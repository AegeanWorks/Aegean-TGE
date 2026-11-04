// Vulkan GPU backend (real implementation).
//
// Compiled only when find_package(Vulkan) succeeds (TGE_HAS_VULKAN=1).
// Owns init()/shutdown()/status()/info()/use_gpu()/run_noop(); noop.cpp
// contributes nothing in that configuration. Uses core Vulkan 1.1 only:
// instance, device, memory, descriptors, compute pipeline, queue, fence.
// No surface, no swapchain, no validation layers: every call's VkResult is
// checked and any failure lands on Unavailable / {false}, never an abort.
//
// Device choice: first discrete GPU, preferring AMD (this box); minimum bar
// is a compute-capable queue family. VRAM sizing against the model plan
// stays with the caller (Qwen2::load), which owns the mmap weights.

#include "tge/gpu/backend.hpp"

#ifdef TGE_HAS_VULKAN

#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "tge_gpu_noop_spv.h"
#include "tge_gpu_q8_gemv_spv.h"
#include "tge_gpu_q8_gemm_spv.h"

namespace tge::gpu {
namespace {

constexpr std::uint64_t kFenceTimeoutNs = 5000000000ULL;

GpuStatus g_status = GpuStatus::Disabled;
GpuInfo g_info;
VkInstance g_instance = nullptr;
VkPhysicalDevice g_phys = nullptr;
VkDevice g_device = nullptr;
VkQueue g_queue = nullptr;
std::uint32_t g_family = 0;
VkCommandPool g_pool = nullptr;

// Persistent offload state, declared here so teardown releases everything
// before the device is destroyed. Weights upload cache: one device-local
// buffer per (mmap pointer, bytes) key, so a prefill chunk pays each of
// the 7-per-layer matrices' upload once no matter how many calls share
// them, and the decode output matrix stays resident across tokens. The key
// is the mmap pointer, which changes on every model reload. Bounded by the
// model's own matrices (a few hundred entries); lookup is linear.
struct ResidentEntry {
  const void* key = nullptr;
  std::size_t bytes = 0;
  VkBuffer buffer = nullptr;
  VkDeviceMemory memory = nullptr;
};
std::vector<ResidentEntry> g_residents;
VkDescriptorSetLayout g_gemv_layout = nullptr;
VkPipelineLayout g_gemv_pipe_layout = nullptr;
VkPipeline g_gemv_pipe = nullptr;
VkDescriptorSetLayout g_gemm_layout = nullptr;
VkPipelineLayout g_gemm_pipe_layout = nullptr;
VkPipeline g_gemm_pipe = nullptr;

void drop_weights() {
  for (ResidentEntry& entry : g_residents) {
    if (entry.buffer != nullptr) {
      vkDestroyBuffer(g_device, entry.buffer, nullptr);
    }
    if (entry.memory != nullptr) {
      vkFreeMemory(g_device, entry.memory, nullptr);
    }
  }
  g_residents.clear();
}

void drop_gemv_pipeline() {
  if (g_gemv_pipe != nullptr) {
    vkDestroyPipeline(g_device, g_gemv_pipe, nullptr);
    g_gemv_pipe = nullptr;
  }
  if (g_gemv_pipe_layout != nullptr) {
    vkDestroyPipelineLayout(g_device, g_gemv_pipe_layout, nullptr);
    g_gemv_pipe_layout = nullptr;
  }
  if (g_gemv_layout != nullptr) {
    vkDestroyDescriptorSetLayout(g_device, g_gemv_layout, nullptr);
    g_gemv_layout = nullptr;
  }
}

void drop_gemm_pipeline() {
  if (g_gemm_pipe != nullptr) {
    vkDestroyPipeline(g_device, g_gemm_pipe, nullptr);
    g_gemm_pipe = nullptr;
  }
  if (g_gemm_pipe_layout != nullptr) {
    vkDestroyPipelineLayout(g_device, g_gemm_pipe_layout, nullptr);
    g_gemm_pipe_layout = nullptr;
  }
  if (g_gemm_layout != nullptr) {
    vkDestroyDescriptorSetLayout(g_device, g_gemm_layout, nullptr);
    g_gemm_layout = nullptr;
  }
}

// Persistent per-call resources. Descriptor sets, command buffer and fence
// are reset and re-recorded every dispatch; the I/O buffers are host-visible
// and stay mapped, growing only when a larger call arrives (decode 0.6 MB
// vs 64-token batch 39 MB). This removes the per-dispatch allocator/fence
// churn that dominated the first profiled slice.
VkDescriptorPool g_io_pool = nullptr;
VkBuffer g_io_in = nullptr;
VkDeviceMemory g_io_in_mem = nullptr;
void* g_io_in_map = nullptr;
VkDeviceSize g_io_in_cap = 0;
VkBuffer g_io_out = nullptr;
VkDeviceMemory g_io_out_mem = nullptr;
void* g_io_out_map = nullptr;
VkDeviceSize g_io_out_cap = 0;
VkCommandBuffer g_cmd = nullptr;
VkFence g_fence = nullptr;

void drop_io() {
  if (g_cmd != nullptr) {
    vkFreeCommandBuffers(g_device, g_pool, 1, &g_cmd);
    g_cmd = nullptr;
  }
  if (g_fence != nullptr) {
    vkDestroyFence(g_device, g_fence, nullptr);
    g_fence = nullptr;
  }
  if (g_io_pool != nullptr) {
    vkDestroyDescriptorPool(g_device, g_io_pool, nullptr);
    g_io_pool = nullptr;
  }
  if (g_io_in_map != nullptr) {
    vkUnmapMemory(g_device, g_io_in_mem);
    g_io_in_map = nullptr;
  }
  if (g_io_in_mem != nullptr) {
    vkFreeMemory(g_device, g_io_in_mem, nullptr);
    g_io_in_mem = nullptr;
  }
  if (g_io_in != nullptr) {
    vkDestroyBuffer(g_device, g_io_in, nullptr);
    g_io_in = nullptr;
  }
  g_io_in_cap = 0;
  if (g_io_out_map != nullptr) {
    vkUnmapMemory(g_device, g_io_out_mem);
    g_io_out_map = nullptr;
  }
  if (g_io_out_mem != nullptr) {
    vkFreeMemory(g_device, g_io_out_mem, nullptr);
    g_io_out_mem = nullptr;
  }
  if (g_io_out != nullptr) {
    vkDestroyBuffer(g_device, g_io_out, nullptr);
    g_io_out = nullptr;
  }
  g_io_out_cap = 0;
}

bool find_mem_type(std::uint32_t bits, VkMemoryPropertyFlags flags,
                   std::uint32_t& type) {
  VkPhysicalDeviceMemoryProperties props{};
  vkGetPhysicalDeviceMemoryProperties(g_phys, &props);
  for (std::uint32_t index = 0; index < props.memoryTypeCount; ++index) {
    if ((bits & (1u << index)) == 0u) {
      continue;
    }
    if ((props.memoryTypes[index].propertyFlags & flags) == flags) {
      type = index;
      return true;
    }
  }
  return false;
}

bool grow_io_buffer(VkBuffer& buf, VkDeviceMemory& mem, void*& map,
                    VkDeviceSize& cap, VkDeviceSize need) {
  if (need <= cap) {
    return true;
  }
  if (map != nullptr) {
    vkUnmapMemory(g_device, mem);
    map = nullptr;
  }
  if (mem != nullptr) {
    vkFreeMemory(g_device, mem, nullptr);
    mem = nullptr;
  }
  if (buf != nullptr) {
    vkDestroyBuffer(g_device, buf, nullptr);
    buf = nullptr;
  }
  cap = 0;
  VkBufferCreateInfo buf_info{};
  buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buf_info.size = need;
  buf_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(g_device, &buf_info, nullptr, &buf) != VK_SUCCESS) {
    return false;
  }
  VkMemoryRequirements req{};
  vkGetBufferMemoryRequirements(g_device, buf, &req);
  std::uint32_t type = 0;
  if (!find_mem_type(req.memoryTypeBits,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     type)) {
    return false;
  }
  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = req.size;
  alloc_info.memoryTypeIndex = type;
  if (vkAllocateMemory(g_device, &alloc_info, nullptr, &mem) != VK_SUCCESS) {
    return false;
  }
  if (vkBindBufferMemory(g_device, buf, mem, 0) != VK_SUCCESS) {
    return false;
  }
  if (vkMapMemory(g_device, mem, 0, need, 0, &map) != VK_SUCCESS) {
    return false;
  }
  cap = need;
  return true;
}

bool ensure_io(VkDeviceSize in_need, VkDeviceSize out_need) {
  if (!grow_io_buffer(g_io_in, g_io_in_mem, g_io_in_map, g_io_in_cap,
                      in_need)) {
    return false;
  }
  if (!grow_io_buffer(g_io_out, g_io_out_mem, g_io_out_map, g_io_out_cap,
                      out_need)) {
    return false;
  }
  if (g_io_pool == nullptr) {
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 3;
    VkDescriptorPoolCreateInfo pool_create{};
    pool_create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_create.maxSets = 1;
    pool_create.poolSizeCount = 1;
    pool_create.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(g_device, &pool_create, nullptr, &g_io_pool) !=
        VK_SUCCESS) {
      return false;
    }
  }
  if (g_fence == nullptr) {
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(g_device, &fence_info, nullptr, &g_fence) != VK_SUCCESS) {
      return false;
    }
  }
  if (g_cmd == nullptr) {
    VkCommandBufferAllocateInfo cmd_alloc{};
    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = g_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(g_device, &cmd_alloc, &g_cmd) != VK_SUCCESS) {
      return false;
    }
  }
  return true;
}

bool has_compute_queue(VkPhysicalDevice device, std::uint32_t& family) {
  std::uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
  if (count == 0) {
    return false;
  }
  // Small static cap: devices with >8 families are unheard of; the loop is
  // bounded and never allocates.
  VkQueueFamilyProperties families[8];
  std::uint32_t query = count < 8 ? count : static_cast<std::uint32_t>(8);
  // The call writes back min(input, available); families holds 8, so the
  // clamped query can never overflow it.
  vkGetPhysicalDeviceQueueFamilyProperties(device, &query, families);
  for (std::uint32_t index = 0; index < query; ++index) {
    if ((families[index].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0U) {
      family = index;
      return true;
    }
  }
  return false;
}

void fill_heap_sizes(VkPhysicalDevice device, GpuInfo& out) {
  VkPhysicalDeviceMemoryProperties memory{};
  vkGetPhysicalDeviceMemoryProperties(device, &memory);
  std::uint64_t device_local = 0;
  std::uint64_t host_visible = 0;
  for (std::uint32_t heap = 0; heap < memory.memoryHeapCount; ++heap) {
    const VkMemoryHeap& entry = memory.memoryHeaps[heap];
    if ((entry.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0U) {
      if (entry.size > device_local) {
        device_local = entry.size;
      }
    } else {
      if (entry.size > host_visible) {
        host_visible = entry.size;
      }
    }
  }
  out.device_local_bytes = device_local;
  out.host_visible_bytes = host_visible;
}

void teardown() {
  // Persistent offload state (upload cache, pipelines, per-call I/O) must
  // die before the device does.
  drop_io();
  drop_gemm_pipeline();
  drop_weights();
  drop_gemv_pipeline();
  if (g_pool != nullptr) {
    vkDestroyCommandPool(g_device, g_pool, nullptr);
    g_pool = nullptr;
  }
  if (g_device != nullptr) {
    vkDestroyDevice(g_device, nullptr);
    g_device = nullptr;
    g_queue = nullptr;
  }
  if (g_instance != nullptr) {
    vkDestroyInstance(g_instance, nullptr);
    g_instance = nullptr;
    g_phys = nullptr;
  }
}

}  // namespace

GpuStatus init() {
  if (g_status != GpuStatus::Disabled) {
    return g_status;
  }
  g_info = GpuInfo{};

  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "aegean-tge";
  app.apiVersion = VK_API_VERSION_1_1;

  VkInstanceCreateInfo create{};
  create.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create.pApplicationInfo = &app;

  if (vkCreateInstance(&create, nullptr, &g_instance) != VK_SUCCESS) {
    g_status = GpuStatus::Unavailable;
    std::strncpy(g_info.name, "vkCreateInstance failed",
                 sizeof(g_info.name) - 1);
    return g_status;
  }

  std::uint32_t device_count = 0;
  if (vkEnumeratePhysicalDevices(g_instance, &device_count, nullptr) !=
          VK_SUCCESS ||
      device_count == 0) {
    teardown();
    g_status = GpuStatus::Unavailable;
    std::strncpy(g_info.name, "no physical devices",
                 sizeof(g_info.name) - 1);
    return g_status;
  }

  // Bounded static buffer: more than 8 GPUs on one box is not a target.
  VkPhysicalDevice devices[8];
  std::uint32_t query =
      device_count < 8 ? device_count : static_cast<std::uint32_t>(8);
  if (vkEnumeratePhysicalDevices(g_instance, &query, devices) != VK_SUCCESS) {
    teardown();
    g_status = GpuStatus::Unavailable;
    std::strncpy(g_info.name, "device enumeration failed",
                 sizeof(g_info.name) - 1);
    return g_status;
  }

  int chosen = -1;
  int first_discrete = -1;
  VkPhysicalDeviceProperties chosen_props{};
  std::uint32_t chosen_family = 0;
  for (std::uint32_t index = 0; index < query; ++index) {
    std::uint32_t family = 0;
    if (!has_compute_queue(devices[index], family)) {
      continue;
    }
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(devices[index], &props);
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
        first_discrete < 0) {
      first_discrete = static_cast<int>(index);
    }
    // Prefer AMD discrete (this box); otherwise first discrete; otherwise
    // the first compute-capable device of any type.
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
        props.vendorID == 0x1002) {
      chosen = static_cast<int>(index);
      chosen_props = props;
      chosen_family = family;
      break;
    }
  }
  if (chosen < 0) {
    chosen = first_discrete >= 0 ? first_discrete : 0;
    if (!has_compute_queue(devices[static_cast<std::uint32_t>(chosen)],
                           chosen_family)) {
      teardown();
      g_status = GpuStatus::Unavailable;
      std::strncpy(g_info.name, "no compute queue", sizeof(g_info.name) - 1);
      return g_status;
    }
    vkGetPhysicalDeviceProperties(devices[static_cast<std::uint32_t>(chosen)],
                                 &chosen_props);
  }
  g_phys = devices[static_cast<std::uint32_t>(chosen)];

  const float priority = 1.0F;
  VkDeviceQueueCreateInfo queue_info{};
  queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_info.queueFamilyIndex = chosen_family;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = &priority;

  VkDeviceCreateInfo device_info{};
  device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos = &queue_info;

  if (vkCreateDevice(g_phys, &device_info, nullptr, &g_device) != VK_SUCCESS) {
    teardown();
    g_status = GpuStatus::Unavailable;
    std::strncpy(g_info.name, "vkCreateDevice failed",
                 sizeof(g_info.name) - 1);
    return g_status;
  }
  vkGetDeviceQueue(g_device, chosen_family, 0, &g_queue);
  g_family = chosen_family;

  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_info.queueFamilyIndex = g_family;
  if (vkCreateCommandPool(g_device, &pool_info, nullptr, &g_pool) !=
      VK_SUCCESS) {
    teardown();
    g_status = GpuStatus::Unavailable;
    std::strncpy(g_info.name, "command pool failed",
                 sizeof(g_info.name) - 1);
    return g_status;
  }

  std::strncpy(g_info.name, chosen_props.deviceName, sizeof(g_info.name) - 1);
  g_info.name[sizeof(g_info.name) - 1] = '\0';
  switch (chosen_props.vendorID) {
    case 0x1002:
      g_info.vendor = GpuVendor::AMD;
      break;
    case 0x10DE:
      g_info.vendor = GpuVendor::NVIDIA;
      break;
    case 0x8086:
      g_info.vendor = GpuVendor::Intel;
      break;
    default:
      g_info.vendor = GpuVendor::Other;
      break;
  }
  fill_heap_sizes(g_phys, g_info);
  g_info.available = g_info.device_local_bytes > 0;
  g_status = g_info.available ? GpuStatus::Ready : GpuStatus::Unavailable;
  if (g_status != GpuStatus::Ready) {
    teardown();
  }
  return g_status;
}

void shutdown() {
  teardown();
  g_status = GpuStatus::Disabled;
  g_info = GpuInfo{};
}

GpuStatus status() { return g_status; }

const GpuInfo& info() { return g_info; }

bool use_gpu() { return g_status == GpuStatus::Ready; }

NoopResult run_noop() {
  NoopResult result;
  if (g_status != GpuStatus::Ready) {
    return result;
  }

  // Every handle below is destroyed on every exit path; failure reports
  // {false} and the caller keeps the CPU kernels.
  VkBuffer buffer = nullptr;
  VkDeviceMemory memory = nullptr;
  void* mapped = nullptr;
  VkDescriptorSetLayout set_layout = nullptr;
  VkDescriptorPool desc_pool = nullptr;
  VkShaderModule module = nullptr;
  VkPipelineLayout pipe_layout = nullptr;
  VkPipeline pipeline = nullptr;
  VkCommandBuffer cmd = nullptr;
  VkFence fence = nullptr;
  bool submitted = false;

  auto fail = [&]() -> NoopResult {
    if (cmd != nullptr) {
      vkFreeCommandBuffers(g_device, g_pool, 1, &cmd);
    }
    if (fence != nullptr) {
      vkDestroyFence(g_device, fence, nullptr);
    }
    if (pipeline != nullptr) {
      vkDestroyPipeline(g_device, pipeline, nullptr);
    }
    if (pipe_layout != nullptr) {
      vkDestroyPipelineLayout(g_device, pipe_layout, nullptr);
    }
    if (module != nullptr) {
      vkDestroyShaderModule(g_device, module, nullptr);
    }
    if (desc_pool != nullptr) {
      vkDestroyDescriptorPool(g_device, desc_pool, nullptr);
    }
    if (set_layout != nullptr) {
      vkDestroyDescriptorSetLayout(g_device, set_layout, nullptr);
    }
    if (mapped != nullptr) {
      vkUnmapMemory(g_device, memory);
    }
    if (memory != nullptr) {
      vkFreeMemory(g_device, memory, nullptr);
    }
    if (buffer != nullptr) {
      vkDestroyBuffer(g_device, buffer, nullptr);
    }
    return result;
  };

  VkBufferCreateInfo buf_info{};
  buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buf_info.size = sizeof(std::uint32_t);
  buf_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(g_device, &buf_info, nullptr, &buffer) != VK_SUCCESS) {
    return fail();
  }
  VkMemoryRequirements req{};
  vkGetBufferMemoryRequirements(g_device, buffer, &req);
  VkPhysicalDeviceMemoryProperties mem_props{};
  vkGetPhysicalDeviceMemoryProperties(g_phys, &mem_props);
  std::uint32_t mem_type = 0xFFFFFFFFu;
  for (std::uint32_t index = 0; index < mem_props.memoryTypeCount; ++index) {
    if ((req.memoryTypeBits & (1u << index)) == 0u) {
      continue;
    }
    const VkMemoryPropertyFlags flags =
        mem_props.memoryTypes[index].propertyFlags;
    if ((flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
        (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
      mem_type = index;
      break;
    }
  }
  if (mem_type == 0xFFFFFFFFu) {
    return fail();
  }
  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = req.size;
  alloc_info.memoryTypeIndex = mem_type;
  if (vkAllocateMemory(g_device, &alloc_info, nullptr, &memory) != VK_SUCCESS) {
    return fail();
  }
  if (vkBindBufferMemory(g_device, buffer, memory, 0) != VK_SUCCESS) {
    return fail();
  }
  if (vkMapMemory(g_device, memory, 0, sizeof(std::uint32_t), 0, &mapped) !=
      VK_SUCCESS) {
    return fail();
  }
  *static_cast<std::uint32_t*>(mapped) = 0u;

  VkDescriptorSetLayoutBinding binding{};
  binding.binding = 0;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  binding.descriptorCount = 1;
  binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = 1;
  layout_info.pBindings = &binding;
  if (vkCreateDescriptorSetLayout(g_device, &layout_info, nullptr,
                                  &set_layout) != VK_SUCCESS) {
    return fail();
  }
  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_size.descriptorCount = 1;
  VkDescriptorPoolCreateInfo pool_create{};
  pool_create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_create.maxSets = 1;
  pool_create.poolSizeCount = 1;
  pool_create.pPoolSizes = &pool_size;
  if (vkCreateDescriptorPool(g_device, &pool_create, nullptr, &desc_pool) !=
      VK_SUCCESS) {
    return fail();
  }
  VkDescriptorSetAllocateInfo set_alloc{};
  set_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  set_alloc.descriptorPool = desc_pool;
  set_alloc.descriptorSetCount = 1;
  set_alloc.pSetLayouts = &set_layout;
  VkDescriptorSet set = nullptr;
  if (vkAllocateDescriptorSets(g_device, &set_alloc, &set) != VK_SUCCESS) {
    return fail();
  }
  VkDescriptorBufferInfo buf_desc{};
  buf_desc.buffer = buffer;
  buf_desc.offset = 0;
  buf_desc.range = sizeof(std::uint32_t);
  VkWriteDescriptorSet write{};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet = set;
  write.dstBinding = 0;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write.pBufferInfo = &buf_desc;
  vkUpdateDescriptorSets(g_device, 1, &write, 0, nullptr);

  VkShaderModuleCreateInfo mod_info{};
  mod_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  mod_info.codeSize = spv::kNoopSpv_bytes;
  mod_info.pCode =
      reinterpret_cast<const std::uint32_t*>(spv::kNoopSpv);
  if (vkCreateShaderModule(g_device, &mod_info, nullptr, &module) !=
      VK_SUCCESS) {
    return fail();
  }
  VkPipelineLayoutCreateInfo pipe_layout_info{};
  pipe_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipe_layout_info.setLayoutCount = 1;
  pipe_layout_info.pSetLayouts = &set_layout;
  if (vkCreatePipelineLayout(g_device, &pipe_layout_info, nullptr,
                             &pipe_layout) != VK_SUCCESS) {
    return fail();
  }
  VkComputePipelineCreateInfo pipe_info{};
  pipe_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipe_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipe_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipe_info.stage.module = module;
  pipe_info.stage.pName = "main";
  pipe_info.layout = pipe_layout;
  if (vkCreateComputePipelines(g_device, nullptr, 1, &pipe_info, nullptr,
                               &pipeline) != VK_SUCCESS) {
    return fail();
  }

  VkCommandBufferAllocateInfo cmd_alloc{};
  cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmd_alloc.commandPool = g_pool;
  cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmd_alloc.commandBufferCount = 1;
  if (vkAllocateCommandBuffers(g_device, &cmd_alloc, &cmd) != VK_SUCCESS) {
    return fail();
  }
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(cmd, &begin) != VK_SUCCESS) {
    return fail();
  }
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout, 0,
                          1, &set, 0, nullptr);
  vkCmdDispatch(cmd, 1, 1, 1);
  if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
    return fail();
  }

  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  if (vkCreateFence(g_device, &fence_info, nullptr, &fence) != VK_SUCCESS) {
    return fail();
  }
  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;
  const auto t0 = std::chrono::steady_clock::now();
  if (vkQueueSubmit(g_queue, 1, &submit, fence) != VK_SUCCESS) {
    return fail();
  }
  submitted = true;
  const VkResult wait =
      vkWaitForFences(g_device, 1, &fence, VK_TRUE, kFenceTimeoutNs);
  const auto t1 = std::chrono::steady_clock::now();
  result.submit_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
  if (wait != VK_SUCCESS) {
    return fail();
  }
  result.readback = *static_cast<const std::uint32_t*>(mapped);
  result.ok = (result.readback == 1u);

  vkFreeCommandBuffers(g_device, g_pool, 1, &cmd);
  cmd = nullptr;
  vkDestroyFence(g_device, fence, nullptr);
  fence = nullptr;
  vkDestroyPipeline(g_device, pipeline, nullptr);
  pipeline = nullptr;
  vkDestroyPipelineLayout(g_device, pipe_layout, nullptr);
  pipe_layout = nullptr;
  vkDestroyShaderModule(g_device, module, nullptr);
  module = nullptr;
  vkDestroyDescriptorPool(g_device, desc_pool, nullptr);
  desc_pool = nullptr;
  vkDestroyDescriptorSetLayout(g_device, set_layout, nullptr);
  set_layout = nullptr;
  vkUnmapMemory(g_device, memory);
  mapped = nullptr;
  vkFreeMemory(g_device, memory, nullptr);
  memory = nullptr;
  vkDestroyBuffer(g_device, buffer, nullptr);
  buffer = nullptr;
  (void)submitted;
  return result;
}

namespace {

// (Persistent weights/pipeline/io state and its drop functions live in the
// first anonymous block above so teardown sees them.)

// One-shot copy on the shared compute queue. Used for the initial weight
// upload only; the per-token path never copies weights.
bool copy_buffer_once(VkBuffer dst, VkBuffer src, VkDeviceSize size) {
  VkCommandBufferAllocateInfo alloc{};
  alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc.commandPool = g_pool;
  alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc.commandBufferCount = 1;
  VkCommandBuffer cmd = nullptr;
  if (vkAllocateCommandBuffers(g_device, &alloc, &cmd) != VK_SUCCESS) {
    return false;
  }
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  bool ok = vkBeginCommandBuffer(cmd, &begin) == VK_SUCCESS;
  if (ok) {
    VkBufferCopy region{};
    region.size = size;
    vkCmdCopyBuffer(cmd, src, dst, 1, &region);
    ok = vkEndCommandBuffer(cmd) == VK_SUCCESS;
  }
  VkFence fence = nullptr;
  if (ok) {
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    ok = vkCreateFence(g_device, &fence_info, nullptr, &fence) == VK_SUCCESS;
  }
  if (ok) {
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    ok = vkQueueSubmit(g_queue, 1, &submit, fence) == VK_SUCCESS;
  }
  if (ok) {
    ok = vkWaitForFences(g_device, 1, &fence, VK_TRUE, kFenceTimeoutNs) ==
         VK_SUCCESS;
  }
  if (fence != nullptr) {
    vkDestroyFence(g_device, fence, nullptr);
  }
  vkFreeCommandBuffers(g_device, g_pool, 1, &cmd);
  return ok;
}

bool ensure_resident(const void* data, std::size_t bytes,
                     std::uint64_t& h2d_ns, VkBuffer& device_out,
                     VkDeviceSize& device_size_out) {
  for (const ResidentEntry& entry : g_residents) {
    if (entry.key == data && entry.bytes == bytes) {
      device_out = entry.buffer;
      device_size_out =
          static_cast<VkDeviceSize>(entry.bytes + 4);  // zero pad, below
      h2d_ns = 0;
      return true;
    }
  }
  const auto t0 = std::chrono::steady_clock::now();

  // The shader assembles words from any byte address, so a block-end read
  // can touch the word past the last weight byte. Pad the upload with one
  // zero word; the key stays the exact (pointer, bytes).
  const std::size_t padded = bytes + 4;

  VkBuffer staging = nullptr;
  VkDeviceMemory staging_mem = nullptr;
  void* mapped = nullptr;
  bool ok = false;
  do {
    VkBufferCreateInfo buf_info{};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.size = static_cast<VkDeviceSize>(padded);
    buf_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(g_device, &buf_info, nullptr, &staging) != VK_SUCCESS) {
      break;
    }
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(g_device, staging, &req);
    std::uint32_t type = 0;
    if (!find_mem_type(req.memoryTypeBits,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       type)) {
      break;
    }
    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = req.size;
    alloc_info.memoryTypeIndex = type;
    if (vkAllocateMemory(g_device, &alloc_info, nullptr, &staging_mem) !=
        VK_SUCCESS) {
      break;
    }
    if (vkBindBufferMemory(g_device, staging, staging_mem, 0) != VK_SUCCESS) {
      break;
    }
    if (vkMapMemory(g_device, staging_mem, 0, static_cast<VkDeviceSize>(padded),
                    0, &mapped) != VK_SUCCESS) {
      break;
    }
    std::memcpy(mapped, data, bytes);
    std::memset(static_cast<std::uint8_t*>(mapped) + bytes, 0, 4);
    vkUnmapMemory(g_device, staging_mem);
    mapped = nullptr;

    VkBuffer device = nullptr;
    VkDeviceMemory device_mem = nullptr;
    buf_info.usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(g_device, &buf_info, nullptr, &device) != VK_SUCCESS) {
      break;
    }
    vkGetBufferMemoryRequirements(g_device, device, &req);
    if (!find_mem_type(req.memoryTypeBits,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, type)) {
      vkDestroyBuffer(g_device, device, nullptr);
      break;
    }
    alloc_info.allocationSize = req.size;
    alloc_info.memoryTypeIndex = type;
    if (vkAllocateMemory(g_device, &alloc_info, nullptr, &device_mem) !=
        VK_SUCCESS) {
      vkDestroyBuffer(g_device, device, nullptr);
      break;
    }
    if (vkBindBufferMemory(g_device, device, device_mem, 0) != VK_SUCCESS) {
      vkFreeMemory(g_device, device_mem, nullptr);
      vkDestroyBuffer(g_device, device, nullptr);
      break;
    }
    if (!copy_buffer_once(device, staging,
                          static_cast<VkDeviceSize>(padded))) {
      vkFreeMemory(g_device, device_mem, nullptr);
      vkDestroyBuffer(g_device, device, nullptr);
      break;
    }
    ResidentEntry entry;
    entry.key = data;
    entry.bytes = bytes;
    entry.buffer = device;
    entry.memory = device_mem;
    g_residents.push_back(entry);
    device_out = device;
    device_size_out = static_cast<VkDeviceSize>(padded);
    ok = true;
  } while (false);

  if (staging != nullptr) {
    vkDestroyBuffer(g_device, staging, nullptr);
  }
  if (staging_mem != nullptr) {
    vkFreeMemory(g_device, staging_mem, nullptr);
  }
  const auto t1 = std::chrono::steady_clock::now();
  h2d_ns = ok ? static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
                        .count())
              : 0;
  return ok;
}

bool ensure_gemv_pipeline() {
  if (g_gemv_pipe != nullptr) {
    return true;
  }
  VkDescriptorSetLayoutBinding bindings[3] = {};
  for (int index = 0; index < 3; ++index) {
    bindings[index].binding = static_cast<std::uint32_t>(index);
    bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[index].descriptorCount = 1;
    bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = 3;
  layout_info.pBindings = bindings;
  if (vkCreateDescriptorSetLayout(g_device, &layout_info, nullptr,
                                  &g_gemv_layout) != VK_SUCCESS) {
    return false;
  }
  VkPushConstantRange push{};
  push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push.offset = 0;
  push.size = 12;
  VkPipelineLayoutCreateInfo pipe_layout_info{};
  pipe_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipe_layout_info.setLayoutCount = 1;
  pipe_layout_info.pSetLayouts = &g_gemv_layout;
  pipe_layout_info.pushConstantRangeCount = 1;
  pipe_layout_info.pPushConstantRanges = &push;
  if (vkCreatePipelineLayout(g_device, &pipe_layout_info, nullptr,
                             &g_gemv_pipe_layout) != VK_SUCCESS) {
    drop_gemv_pipeline();
    return false;
  }
  VkShaderModule mod = nullptr;
  VkShaderModuleCreateInfo mod_info{};
  mod_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  mod_info.codeSize = spv::kQ8GemvSpv_bytes;
  mod_info.pCode = reinterpret_cast<const std::uint32_t*>(spv::kQ8GemvSpv);
  if (vkCreateShaderModule(g_device, &mod_info, nullptr, &mod) != VK_SUCCESS) {
    drop_gemv_pipeline();
    return false;
  }
  VkComputePipelineCreateInfo pipe_info{};
  pipe_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipe_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipe_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipe_info.stage.module = mod;
  pipe_info.stage.pName = "main";
  pipe_info.layout = g_gemv_pipe_layout;
  const bool ok = vkCreateComputePipelines(g_device, nullptr, 1, &pipe_info,
                                           nullptr, &g_gemv_pipe) == VK_SUCCESS;
  vkDestroyShaderModule(g_device, mod, nullptr);
  if (!ok) {
    drop_gemv_pipeline();
  }
  return ok;
}

}  // namespace

// Shared per-dispatch path for both Q8 kernels. I/O buffers, descriptor
// pool, command buffer and fence persist across calls (grown/reset, never
// reallocated in steady state); only the descriptor set itself is fresh per
// call. Weights stay resident via the upload cache. Times split host-side
// setup (alloc_ns) from queue time (submit_ns) so the log can tell
// allocator churn apart from kernel time.
static bool run_q8_call(VkPipeline pipe, VkPipelineLayout pipe_layout,
                        VkDescriptorSetLayout set_layout, VkBuffer weights,
                        VkDeviceSize weights_size, const void* input,
                        std::size_t in_bytes, void* output,
                        std::size_t out_bytes, const std::uint32_t* push,
                        std::size_t push_bytes, std::uint32_t groups_x,
                        std::uint32_t groups_y, GemvQ8Result& result) {
  const auto ta0 = std::chrono::steady_clock::now();
  if (!ensure_io(static_cast<VkDeviceSize>(in_bytes),
                 static_cast<VkDeviceSize>(out_bytes))) {
    return false;
  }
  vkResetDescriptorPool(g_device, g_io_pool, 0);
  vkResetCommandBuffer(g_cmd, 0);
  vkResetFences(g_device, 1, &g_fence);
  VkDescriptorSet set = nullptr;
  VkDescriptorSetAllocateInfo set_alloc{};
  set_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  set_alloc.descriptorPool = g_io_pool;
  set_alloc.descriptorSetCount = 1;
  set_alloc.pSetLayouts = &set_layout;
  if (vkAllocateDescriptorSets(g_device, &set_alloc, &set) != VK_SUCCESS) {
    return false;
  }
  VkDescriptorBufferInfo buf_infos[3] = {};
  buf_infos[0].buffer = weights;
  buf_infos[0].offset = 0;
  buf_infos[0].range = weights_size;
  buf_infos[1].buffer = g_io_in;
  buf_infos[1].offset = 0;
  buf_infos[1].range = static_cast<VkDeviceSize>(in_bytes);
  buf_infos[2].buffer = g_io_out;
  buf_infos[2].offset = 0;
  buf_infos[2].range = static_cast<VkDeviceSize>(out_bytes);
  VkWriteDescriptorSet writes[3] = {};
  for (int index = 0; index < 3; ++index) {
    writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[index].dstSet = set;
    writes[index].dstBinding = static_cast<std::uint32_t>(index);
    writes[index].descriptorCount = 1;
    writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[index].pBufferInfo = &buf_infos[index];
  }
  vkUpdateDescriptorSets(g_device, 3, writes, 0, nullptr);
  std::memcpy(g_io_in_map, input, in_bytes);
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(g_cmd, &begin) != VK_SUCCESS) {
    return false;
  }
  vkCmdBindPipeline(g_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  vkCmdBindDescriptorSets(g_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout,
                          0, 1, &set, 0, nullptr);
  vkCmdPushConstants(g_cmd, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     push_bytes, push);
  vkCmdDispatch(g_cmd, groups_x, groups_y, 1);
  if (vkEndCommandBuffer(g_cmd) != VK_SUCCESS) {
    return false;
  }
  const auto ta1 = std::chrono::steady_clock::now();
  result.alloc_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(ta1 - ta0).count());
  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &g_cmd;
  const auto t0 = std::chrono::steady_clock::now();
  if (vkQueueSubmit(g_queue, 1, &submit, g_fence) != VK_SUCCESS) {
    return false;
  }
  const VkResult wait =
      vkWaitForFences(g_device, 1, &g_fence, VK_TRUE, kFenceTimeoutNs);
  const auto t1 = std::chrono::steady_clock::now();
  result.submit_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
  if (wait != VK_SUCCESS) {
    return false;
  }
  const auto t2 = std::chrono::steady_clock::now();
  std::memcpy(output, g_io_out_map, out_bytes);
  const auto t3 = std::chrono::steady_clock::now();
  result.d2h_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count());
  result.ok = true;
  if (std::getenv("TGE_GPU_TRACE") != nullptr) {
    std::fprintf(stderr,
                 "gpu trace: alloc %.2f ms, h2d %.2f ms, submit %.2f ms, "
                 "d2h %.2f ms\n",
                 static_cast<double>(result.alloc_ns) / 1e6,
                 static_cast<double>(result.h2d_ns) / 1e6,
                 static_cast<double>(result.submit_ns) / 1e6,
                 static_cast<double>(result.d2h_ns) / 1e6);
  }
  return true;
}

GemvQ8Result gemv_q8(const void* weights, std::size_t weight_bytes, int rows,
                     int cols, std::size_t row_stride_bytes,
                     const float* input, float* output) {
  GemvQ8Result result;
  if (g_status != GpuStatus::Ready) {
    return result;
  }
  if (weights == nullptr || input == nullptr || output == nullptr ||
      rows <= 0 || cols <= 0 || (cols % 32) != 0 || row_stride_bytes == 0) {
    return result;
  }
  const std::uint32_t blocks = static_cast<std::uint32_t>(cols) / 32u;
  if (row_stride_bytes < static_cast<std::size_t>(blocks) * 34u) {
    return result;
  }
  const std::size_t in_bytes = static_cast<std::size_t>(cols) * sizeof(float);
  const std::size_t out_bytes = static_cast<std::size_t>(rows) * sizeof(float);
  VkBuffer resident = nullptr;
  VkDeviceSize resident_size = 0;
  if (!ensure_resident(weights, weight_bytes, result.h2d_ns, resident,
                       resident_size)) {
    return result;
  }
  if (!ensure_gemv_pipeline()) {
    return result;
  }
  const std::uint32_t params[3] = {static_cast<std::uint32_t>(rows),
                                   static_cast<std::uint32_t>(row_stride_bytes),
                                   blocks};
  run_q8_call(g_gemv_pipe, g_gemv_pipe_layout, g_gemv_layout, resident,
              resident_size, input, in_bytes, output, out_bytes, params,
              sizeof(params),
              (static_cast<std::uint32_t>(rows) + 63u) / 64u, 1, result);
  return result;
}

bool ensure_q8gemm_pipeline() {
  if (g_gemm_pipe != nullptr) {
    return true;
  }
  VkDescriptorSetLayoutBinding bindings[3] = {};
  for (int index = 0; index < 3; ++index) {
    bindings[index].binding = static_cast<std::uint32_t>(index);
    bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[index].descriptorCount = 1;
    bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = 3;
  layout_info.pBindings = bindings;
  if (vkCreateDescriptorSetLayout(g_device, &layout_info, nullptr,
                                  &g_gemm_layout) != VK_SUCCESS) {
    return false;
  }
  VkPushConstantRange push{};
  push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push.offset = 0;
  push.size = 20;
  VkPipelineLayoutCreateInfo pipe_layout_info{};
  pipe_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipe_layout_info.setLayoutCount = 1;
  pipe_layout_info.pSetLayouts = &g_gemm_layout;
  pipe_layout_info.pushConstantRangeCount = 1;
  pipe_layout_info.pPushConstantRanges = &push;
  if (vkCreatePipelineLayout(g_device, &pipe_layout_info, nullptr,
                             &g_gemm_pipe_layout) != VK_SUCCESS) {
    drop_gemm_pipeline();
    return false;
  }
  VkShaderModule mod = nullptr;
  VkShaderModuleCreateInfo mod_info{};
  mod_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  mod_info.codeSize = spv::kQ8GemmSpv_bytes;
  mod_info.pCode = reinterpret_cast<const std::uint32_t*>(spv::kQ8GemmSpv);
  if (vkCreateShaderModule(g_device, &mod_info, nullptr, &mod) != VK_SUCCESS) {
    drop_gemm_pipeline();
    return false;
  }
  VkComputePipelineCreateInfo pipe_info{};
  pipe_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipe_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipe_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipe_info.stage.module = mod;
  pipe_info.stage.pName = "main";
  pipe_info.layout = g_gemm_pipe_layout;
  const bool ok = vkCreateComputePipelines(g_device, nullptr, 1, &pipe_info,
                                           nullptr, &g_gemm_pipe) == VK_SUCCESS;
  vkDestroyShaderModule(g_device, mod, nullptr);
  if (!ok) {
    drop_gemm_pipeline();
  }
  return ok;
}

GemmQ8Result gemm_q8(const void* weights, std::size_t weight_bytes, int rows,
                     int cols, std::size_t row_stride_bytes,
                     const float* input, float* output, int tokens) {
  GemmQ8Result result;
  if (g_status != GpuStatus::Ready) {
    return result;
  }
  if (weights == nullptr || input == nullptr || output == nullptr ||
      rows <= 0 || cols <= 0 || (cols % 32) != 0 || row_stride_bytes == 0 ||
      tokens <= 0) {
    return result;
  }
  const std::uint32_t blocks =
      static_cast<std::uint32_t>(cols) / 32u;
  if (row_stride_bytes < static_cast<std::size_t>(blocks) * 34u) {
    return result;
  }
  const std::size_t in_bytes = static_cast<std::size_t>(tokens) *
                               static_cast<std::size_t>(cols) * sizeof(float);
  const std::size_t out_bytes = static_cast<std::size_t>(tokens) *
                                static_cast<std::size_t>(rows) * sizeof(float);

  VkBuffer resident = nullptr;
  VkDeviceSize resident_size = 0;
  if (!ensure_resident(weights, weight_bytes, result.h2d_ns, resident,
                       resident_size)) {
    return result;
  }
  if (!ensure_q8gemm_pipeline()) {
    return result;
  }
  const std::uint32_t params[5] = {static_cast<std::uint32_t>(rows),
                                   static_cast<std::uint32_t>(row_stride_bytes),
                                   blocks,
                                   static_cast<std::uint32_t>(cols),
                                   static_cast<std::uint32_t>(tokens)};
  run_q8_call(g_gemm_pipe, g_gemm_pipe_layout, g_gemm_layout, resident,
              resident_size, input, in_bytes, output, out_bytes, params,
              sizeof(params),
              (static_cast<std::uint32_t>(rows) + 7u) / 8u,
              (static_cast<std::uint32_t>(tokens) + 7u) / 8u, result);
  return result;
}

}  // namespace tge::gpu

#endif  // TGE_HAS_VULKAN
