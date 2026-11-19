#pragma once

// Prompt-lookup speculation draft builders. Pure functions over a token
// history: they propose `need` continuations that the verifier still decides.
// Kept header-only so the "all `need` slots are written on success" invariant
// is unit-testable without a model (see review-uninitialized-draft-slot).

#include <algorithm>
#include <cstddef>
#include <vector>

namespace tge::cli {

/// Fills any unwritten draft slots [filled, need) with the last real proposal.
/// A matcher that succeeds on the 5-deep fallback of a 6-deep greedy draft
/// writes only draft[0..4]; without this, draft[5] is uninitialized stack
/// memory and the caller copies garbage into the verifier batch.
inline void complete_draft_tail(int* draft, int filled, int need) {
  for (int index = filled; index < need; ++index) {
    draft[index] = draft[filled - 1];
  }
}

/// Bigram-chain fallback: when no exact span matches, chain the most frequent
/// successor of each token in history. draft[0] follows the last history token;
/// draft[i + 1] follows draft[i]. Proposes only observed continuations.
/// `need` is 6 for the greedy deep draft, 5 for the sampled path.
inline bool bigram_chain_draft(const std::vector<int>& history, int* draft,
                               int need) {
  const int length = static_cast<int>(history.size());
  if (length < 2 || need < 1 || need > 6) {
    return false;
  }
  int prev = history.back();
  for (int index = 0; index < need; ++index) {
    int best = -1;
    int best_count = 0;
    int candidates[32] = {0};
    int tallies[32] = {0};
    int distinct = 0;
    for (int pos = 0; pos < length - 1; ++pos) {
      if (history[static_cast<std::size_t>(pos)] != prev) {
        continue;
      }
      const int cand = history[static_cast<std::size_t>(pos + 1)];
      int slot = 0;
      while (slot < distinct && candidates[slot] != cand) {
        ++slot;
      }
      if (slot == distinct) {
        if (distinct == 32) {
          continue;
        }
        candidates[distinct] = cand;
        tallies[distinct] = 1;
        ++distinct;
      } else {
        ++tallies[slot];
      }
      if (tallies[slot] > best_count) {
        best_count = tallies[slot];
        best = cand;
      }
    }
    if (best < 0) {
      return false;
    }
    draft[index] = best;
    prev = best;
  }
  return true;
}

/// Drafts the continuation of a near-match of the last two to six tokens with
/// exactly one substitution. Last resort after exact spans and bigram chain.
inline bool gapped_draft(const std::vector<int>& history, int* draft, int need) {
  const int length = static_cast<int>(history.size());
  if (need < 1 || need > 6) {
    return false;
  }
  for (int end = length - 2; end >= 1; --end) {
    if (length - 1 - end < need) {
      continue;
    }
    for (int span = 6; span >= 2; --span) {
      if (span > end + 1 || length < span + need) {
        continue;
      }
      int mismatches = 0;
      for (int offset = 0; offset < span; ++offset) {
        if (history[static_cast<std::size_t>(length - span + offset)] !=
            history[static_cast<std::size_t>(end - span + 1 + offset)]) {
          if (++mismatches > 1) {
            break;
          }
        }
      }
      if (mismatches != 1) {
        continue;
      }
      for (int index = 0; index < need; ++index) {
        draft[index] = history[static_cast<std::size_t>(end + 1 + index)];
      }
      return true;
    }
  }
  return false;
}

/// Detects a period-1 (aaaa) or period-2 (abab) lock in recent history and
/// proposes `need` tokens along the locked cycle. Verifier-gated; the caller
/// still applies the d0 pre-check, so a wrong lock costs no weight pass.
inline bool skip_locked_draft(const std::vector<int>& history, int* draft,
                              int need) {
  const int length = static_cast<int>(history.size());
  if (need < 1 || need > 6 || length < 8) {
    return false;
  }
  const int a = history[static_cast<std::size_t>(length - 1)];
  const int b = history[static_cast<std::size_t>(length - 2)];
  bool period1 = true;
  for (int i = 1; i <= 6; ++i) {
    if (history[static_cast<std::size_t>(length - i)] != a) {
      period1 = false;
      break;
    }
  }
  bool period2 = false;
  if (!period1) {
    period2 = true;
    for (int i = 0; i < 8; ++i) {
      const int expect = (i % 2 == 0) ? a : b;
      if (history[static_cast<std::size_t>(length - 1 - i)] != expect) {
        period2 = false;
        break;
      }
    }
  }
  if (!period1 && !period2) {
    return false;
  }
  for (int index = 0; index < need; ++index) {
    if (period1) {
      draft[index] = a;
    } else {
      draft[index] = (index % 2 == 0) ? a : b;
    }
  }
  return true;
}

/// Drafts the continuation of the latest earlier match of the last one to six
/// tokens. Returns `need` tokens or nothing. `need` is 6 on the greedy path and
/// 5 on the sampled path. Greedy tries the deep draft first and falls back to
/// the historical five-deep form. On success ALL `need` slots are written --
/// the five-deep fallback fills the sixth slot via complete_draft_tail.
inline bool prompt_lookup_draft(const std::vector<int>& history, int* draft,
                                int need) {
  if (need < 1 || need > 6) {
    return false;
  }
  const int length = static_cast<int>(history.size());
  for (int pass = 0; pass < 2 && (pass == 0 || need > 5); ++pass) {
    const int want = (pass == 0) ? need : 5;
    for (int span = 6; span >= 1; --span) {
      if (length < span + want) {
        continue;
      }
      const auto suffix_begin = history.begin() + (length - span);
      const auto suffix_end = history.end();
      for (int end = length - 2; end >= span - 1; --end) {
        if (length - 1 - end < want) {
          continue;
        }
        if (!std::equal(suffix_begin, suffix_end,
                        history.begin() + (end - span + 1))) {
          continue;
        }
        for (int index = 0; index < want; ++index) {
          draft[index] = history[static_cast<std::size_t>(end + 1 + index)];
        }
        complete_draft_tail(draft, want, need);
        return true;
      }
    }
    if (bigram_chain_draft(history, draft, want)) {
        complete_draft_tail(draft, want, need);
      return true;
    }
    if (gapped_draft(history, draft, want)) {
        complete_draft_tail(draft, want, need);
      return true;
    }
  }
  return false;
}

}  // namespace tge::cli
