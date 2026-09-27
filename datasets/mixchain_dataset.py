"""MixChain-Z-PRM12K record handling.

The download script already writes {task, step, answer}, so ``data_process`` is a
passthrough. The interesting part is answer checking.

MixChain comes from the math split of PRM800K -- competition problems -- so
targets are NOT all plain integers the way GSM8K's are. They include fractions,
radicals, coordinate pairs and symbolic expressions. The two comparisons the
GSM8K pipeline uses are both wrong here:

    cold start   float(pred) == float(true)   ValueError on "\\frac{1}{2}"
    finetune     pred == true                 "1/2" != "\\frac{1}{2}"

Cold start only KEEPS topologies that solved the task, so a comparison that
under-reports success silently discards good graphs and shrinks the training set
-- the exact scarcity this corpus was chosen to fix. Hence a real normaliser:
``_strip_string`` from gsm8k_dataset is the MATH benchmark's own canonicaliser
(it is where that code came from), applied to both sides, with a numeric
fallback for values that differ only in formatting (2 vs 2.0).
"""
import re

from datasets.gsm8k_dataset import _strip_string


def mixchain_data_process(dataset):
    """Records are already {task, step, answer}; keep every extra field."""
    out = []
    for data in dataset:
        item = dict(data)
        item.setdefault("task", data.get("question", ""))
        item.setdefault("step", data.get("answer", ""))
        out.append(item)
    return out


def _unbox(text: str) -> str:
    """Strip a trailing ``\boxed{...}`` wrapper, if there is one."""
    if "boxed" not in text:
        return text
    tail = text.split("boxed")[-1]
    if tail.startswith("{"):
        depth, buf = 1, ""
        for c in tail[1:]:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            buf += c
        return buf.strip()
    return tail.split("$")[0].strip()


def _extract(pred_str: str) -> str:
    """Pull the asserted answer out of a model response.

    The marker and the box are not alternatives: models routinely write
    "The answer is \boxed{42}", and MathSolver itself appends
    "\nthe answer is {answer}" to a response that may already be boxed. The
    marker branch used to return without unboxing, so every such answer kept its
    "\boxed{...}" wrapper and compared unequal -- silently, and worst on
    symbolic answers, which are the majority of olympiad-style corpora.
    """
    if not isinstance(pred_str, str):
        return ""
    for marker in ("The answer is ", "the answer is "):
        if marker in pred_str:
            return _unbox(pred_str.split(marker)[-1].strip()).strip()
    if "boxed" in pred_str:
        return _unbox(pred_str)
    # Last resort: the final number anywhere in the text.
    nums = re.findall(r"-?\d*\.?\d+", pred_str)
    return nums[-1] if nums else ""


def _mixchain_get_predict_impl(pred_str: str) -> str:
    pred = _extract(pred_str)
    pred = pred.rstrip(".").rstrip("/").strip()
    return _strip_string(pred)


def mixchain_get_predict(pred_str: str) -> str:
    """Total wrapper: a parse failure is a wrong answer, never a crashed run."""
    try:
        return _mixchain_get_predict_impl(pred_str)
    except Exception:
        return ""


def _same(p: str, g: str) -> bool:
    if not p or not g:
        return False
    if p == g:
        return True
    try:                                  # "2" vs "2.0", "0.5" vs ".5"
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        return False


def mixchain_is_correct(pred, gold) -> bool:
    """Normalised comparison, with a numeric fallback.

    Used in place of the GSM8K pipeline's ``float(a) == float(b)`` (cold start)
    and ``a == b`` (finetune), both of which reject correct symbolic answers.

    MUST be idempotent, because both call sites already parse: cold start does
    ``mixchain_is_correct(mixchain_get_predict(raw), gold)``. An earlier version
    re-extracted its first argument, and _extract's last-resort "final number in
    the text" rule then reduced an already-parsed "\\frac{1}{2}" to "2" -- so
    every symbolic answer was graded wrong, silently, for 35% of this corpus.

    So try the input BOTH ways -- as an already-parsed answer and as a raw model
    response -- and accept either.
    """
    try:
        g = _strip_string(str(gold).strip()) if gold is not None else ""
        raw = str(pred).strip() if pred is not None else ""
    except Exception:
        return False
    if not raw or not g:
        return False

    try:
        if _same(_strip_string(raw), g):        # already parsed
            return True
    except Exception:
        pass
    try:
        if _same(mixchain_get_predict(raw), g):  # raw model response
            return True
    except Exception:
        pass
    return False
