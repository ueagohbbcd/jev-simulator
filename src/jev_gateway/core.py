"""Pure request planning, response parsing, and answer aggregation."""
from __future__ import annotations

import json
import math
import re
import string
from collections.abc import Iterable
from typing import Any


class CoreError(ValueError):
    """The upstream evidence is unusable or the inferred plan is inconsistent."""


_SLOTS = re.compile(r"\{\{\s*(state|instructions|options|output)\s*\}\}")
_DEFAULT_CALLSIGNS = list(string.ascii_uppercase)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _render(template: str, values: dict[str, str]) -> str:
    # re.sub evaluates matches from the original template, so inserted evidence
    # containing a placeholder-shaped string is never expanded recursively.
    return _SLOTS.sub(lambda match: values[match.group(1)], template)


def _question_parts(question: dict[str, Any]) -> tuple[list[str], list[str]]:
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "noul":
        criteria = criteria or {}
        return ["true", "false"], [
            _text(criteria.get("true", "Yes")),
            _text(criteria.get("false", "No")),
        ]
    if kind == "choice":
        keys = list(criteria)
        descriptions = [_text(key if criteria[key] is None else criteria[key]) for key in keys]
        return keys, descriptions
    if kind == "score":
        return [str(index) for index in range(len(criteria))], [_text(value) for value in criteria]
    raise CoreError(f"unsupported question type: {kind!r}")


def _branch(
    question_id: str,
    index: int,
    question: dict[str, Any],
    state: Any,
    semantic_keys: list[str],
    descriptions: list[str],
    callsigns: list[str],
    prompt: dict[str, str],
    pair: list[str] | None,
    mode: str = "token_logprobs",
    image_urls: list[str] | None = None,
) -> dict[str, Any]:
    kind = question["type"]
    if kind == "noul":
        tokens = ["Yes", "No"]
    else:
        if len(callsigns) < len(semantic_keys):
            raise CoreError("not enough callsigns for question options")
        tokens = callsigns[:len(semantic_keys)]

    mapping = dict(zip(tokens, semantic_keys, strict=True))
    option_lines = [f"{token}: {description}" for token, description in zip(tokens, descriptions, strict=True)]
    if mode == "reported_probability":
        keys = ", ".join(json.dumps(token, ensure_ascii=False) for token in tokens)
        output = (
            "Output only one JSON object with exactly these keys: " + keys + ". "
            "Each value must be a numeric probability from 0 to 1; "
            "all values must sum to 1. Do not add other keys or text."
        )
    else:
        output = "Output exactly one of: " + ", ".join(tokens) + "."
    values = {
        "state": _text(state),
        "instructions": _text(question["instructions"]),
        "options": "\n".join(option_lines),
        "output": output,
    }
    user_text = _render(prompt["user"], values)
    user_content: str | list[dict[str, Any]] = user_text
    if image_urls:
        user_content = [
            {"type": "image_url", "image_url": {"url": url}} for url in image_urls
        ] + [{"type": "text", "text": user_text}]
    return {
        "question_id": question_id,
        "index": index,
        "mapping": mapping,
        "messages": [
            {"role": "system", "content": _render(prompt["system"], values)},
            {"role": "user", "content": user_content},
        ],
        "pair": pair,
    }


def build_plan(
    payload: dict[str, Any], config: dict[str, Any], *, image_urls: list[str] | None = None
) -> list[dict[str, Any]]:
    """Build all independent completion branches for a validated request."""
    if payload.get("images"):
        raise CoreError("images must be validated before request planning")
    adapter = config.get("adapter", {})
    configured_callsigns = adapter.get("callsigns", _DEFAULT_CALLSIGNS)
    # An empty list disables custom callsigns and restores the stable A-Z set.
    callsigns = list(configured_callsigns) if configured_callsigns else list(_DEFAULT_CALLSIGNS)
    double_round_robin = adapter.get("double_round_robin", False)
    mode = adapter.get("mode", "token_logprobs")
    prompt = config["prompt"]
    result: list[dict[str, Any]] = []

    for question_id, question in payload["questions"].items():
        semantic_keys, descriptions = _question_parts(question)
        if double_round_robin and question["type"] in {"choice", "score"} and len(semantic_keys) >= 2:
            index = 0
            for left in range(len(semantic_keys)):
                for right in range(left + 1, len(semantic_keys)):
                    for first, second in ((left, right), (right, left)):
                        indices = [first, second]
                        pair_keys = [semantic_keys[position] for position in indices]
                        result.append(_branch(
                            question_id, index, question, payload["state"], pair_keys,
                            [descriptions[position] for position in indices], callsigns,
                            prompt, list(pair_keys), mode, image_urls,
                        ))
                        index += 1
        else:
            result.append(_branch(
                question_id, 0, question, payload["state"], semantic_keys,
                descriptions, callsigns, prompt, None, mode, image_urls,
            ))
    return result


def _valid_logprob(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value <= 0
        and value > -9999
    )


def _softmax(logprobs: dict[str, float], temperature: float) -> dict[str, float]:
    peak = max(logprobs.values())
    weights = {
        key: math.exp((value - peak) / temperature)
        for key, value in logprobs.items()
    }
    denominator = math.fsum(weights.values())
    return {key: value / denominator for key, value in weights.items()}


def _warning(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def parse_response(
    response: dict[str, Any],
    mapping: dict[str, str],
    temperature: float,
    low_mass_threshold: float,
) -> dict[str, Any]:
    """Parse first-token logprobs and retain the complete upstream evidence."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise CoreError("temperature must be finite and positive")
    try:
        choice = response["choices"][0]
        row = choice["logprobs"]["content"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise CoreError("no first-output-token logprobs in response") from exc
    if not isinstance(choice, dict) or not isinstance(row, dict):
        raise CoreError("no first-output-token logprobs in response")
    finish_reason = choice.get("finish_reason")
    if finish_reason not in {"stop", "length"}:
        raise CoreError("completion did not finish with stop or length")
    top = row.get("top_logprobs")
    if not isinstance(top, list) or not top:
        raise CoreError("no top-logprobs candidates")

    warnings: list[dict[str, str]] = []
    observed: dict[str, float] = {}
    top_observed: dict[str, float] = {}
    invalid_top = False

    def add(entry: Any, *, is_top: bool) -> None:
        nonlocal invalid_top
        if not isinstance(entry, dict):
            invalid_top |= is_top
            return
        token, logprob = entry.get("token"), entry.get("logprob")
        if not isinstance(token, str) or not _valid_logprob(logprob):
            invalid_top |= is_top
            return
        value = float(logprob)
        previous = observed.get(token)
        if previous is not None:
            if not math.isclose(previous, value, rel_tol=0, abs_tol=1e-6):
                raise CoreError("conflicting logprobs for the same token")
            value = previous
        observed[token] = value
        if is_top:
            top_observed[token] = value

    for entry in top:
        add(entry, is_top=True)
    # The sampled token is evidence when valid, but does not participate in
    # validating the top-k cutoff or protocol argmax.
    add({"token": row.get("token"), "logprob": row.get("logprob")}, is_top=False)
    if invalid_top:
        warnings.append(_warning(
            "invalid_top_token",
            "One or more top-logprobs entries were invalid and were ignored.",
        ))
    if not observed:
        raise CoreError("no usable first-position logprobs")
    if top_observed:
        leader_logprob = max(top_observed.values())
        leaders = [token for token, value in top_observed.items() if value == leader_logprob]
        if any(token not in mapping for token in leaders):
            warnings.append(_warning(
                "non_protocol_leader",
                "A highest-probability top token was not an allowed output token.",
            ))

    label_logprobs = {
        semantic: observed[token]
        for token, semantic in mapping.items()
        if token in observed
    }
    missing = [semantic for token, semantic in mapping.items() if token not in observed]
    if not label_logprobs:
        raise CoreError("none of the allowed output tokens were observed")

    calibrated_observed = _softmax(label_logprobs, temperature)
    raw_observed = _softmax(label_logprobs, 1.0)
    probabilities = {semantic: calibrated_observed.get(semantic, 0.0) for semantic in mapping.values()}
    raw_probabilities = {semantic: raw_observed.get(semantic, 0.0) for semantic in mapping.values()}

    lower = math.fsum(math.exp(value) for value in label_logprobs.values())
    seen_mass = math.fsum(math.exp(value) for value in observed.values())
    if seen_mass > 1.0 + 1e-6:
        raise CoreError("observed token probability mass exceeds one")
    lower = min(1.0, lower)
    tail = max(0.0, 1.0 - seen_mass)
    valid_top_values = [float(entry["logprob"]) for entry in top if (
        isinstance(entry, dict)
        and isinstance(entry.get("token"), str)
        and _valid_logprob(entry.get("logprob"))
    )]
    cutoff = None
    if not invalid_top and len(valid_top_values) == len(top):
        cutoff = min(math.exp(value) for value in valid_top_values)
    extra = tail if cutoff is None else min(tail, len(missing) * cutoff)
    upper = lower if not missing else min(1.0, lower + extra)

    if missing:
        warnings.append(_warning(
            "incomplete_labels",
            "Some allowed output tokens were absent; their probabilities were set to zero.",
        ))
        if temperature != 1:
            warnings.append(_warning(
                "temperature_with_missing",
                "Temperature scaling is conditional on the observed allowed tokens.",
            ))
    if upper < low_mass_threshold:
        warnings.append(_warning(
            "low_label_mass",
            "Even the upper bound on allowed-token probability mass is below the threshold.",
        ))
    elif lower < low_mass_threshold <= upper:
        warnings.append(_warning(
            "uncertain_low_label_mass",
            "Allowed-token probability mass may be below the threshold.",
        ))

    return {
        "probabilities": probabilities,
        "raw_probabilities": raw_probabilities,
        "label_logprobs": label_logprobs,
        "missing_labels": missing,
        "observed_top_tokens": sorted(observed.items(), key=lambda item: -item[1]),
        "observed_label_mass": lower,
        "label_mass_upper_bound": upper,
        "warnings": warnings,
        "finish_reason": finish_reason,
        "sampled_first_token": row.get("token"),
        "model": response.get("model"),
        "system_fingerprint": response.get("system_fingerprint"),
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CoreError("reported probability JSON contains duplicate keys")
        result[key] = value
    return result


def _nonfinite_constant(_: str) -> None:
    raise CoreError("reported probability JSON contains a nonfinite number")


def parse_reported_response(
    response: dict[str, Any], mapping: dict[str, str], temperature: float
) -> dict[str, Any]:
    """Read a complete JSON distribution without inventing token evidence."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise CoreError("temperature must be finite and positive")
    try:
        choices = response["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise CoreError("expected exactly one completion choice")
        choice = choices[0]
        if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
            raise CoreError("reported completion did not finish normally")
        message = choice["message"]
        if not isinstance(message, dict) or message.get("refusal"):
            raise CoreError("reported completion was refused or malformed")
        content = message["content"]
    except (KeyError, TypeError) as exc:
        raise CoreError("reported completion is malformed") from exc
    if not isinstance(content, str):
        raise CoreError("reported completion content must be text")
    try:
        reported = json.loads(
            content, object_pairs_hook=_unique_object, parse_constant=_nonfinite_constant
        )
    except (ValueError, TypeError, RecursionError) as exc:
        raise CoreError("reported completion is not valid JSON") from exc
    if not isinstance(reported, dict) or set(reported) != set(mapping):
        raise CoreError("reported completion must contain exactly the branch labels")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value < 0
        or value > 1
        or not math.isfinite(value)
        for value in reported.values()
    ):
        raise CoreError("reported probabilities must be finite numbers in [0, 1]")
    total = math.fsum(reported.values())
    if abs(total - 1.0) > 1e-6:
        raise CoreError("reported probabilities must sum to one")
    # Accept only tiny numerical drift, then put the vector on the simplex.
    raw = {semantic: float(reported[label]) / total for label, semantic in mapping.items()}
    calibrated = raw if temperature == 1 else _softmax(
        {semantic: math.log(value) if value else -math.inf for semantic, value in raw.items()},
        temperature,
    )
    return {
        "probabilities": calibrated,
        "raw_probabilities": raw,
        "reported_label_probabilities": reported,
        "raw_output": content,
        "normalization_delta": total - 1.0,
        "warnings": [],
        "finish_reason": "stop",
        "model": response.get("model"),
        "system_fingerprint": response.get("system_fingerprint"),
    }


def _semantic_keys(question: dict[str, Any]) -> list[str]:
    return _question_parts(question)[0]


def _confidence(probabilities: Iterable[float]) -> float:
    values = list(probabilities)
    if len(values) <= 1:
        return 1.0
    entropy = -math.fsum(value * math.log(value) for value in values if value > 0)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(len(values))))


def _single_distribution(keys: list[str], runs: list[dict[str, Any]]) -> dict[str, float]:
    if len(runs) != 1:
        raise CoreError("single-question aggregation requires exactly one run")
    source = runs[0].get("probabilities")
    if not isinstance(source, dict):
        raise CoreError("run has no probability distribution")
    result = {key: float(source.get(key, 0.0)) for key in keys}
    total = math.fsum(result.values())
    if not math.isfinite(total) or total <= 0:
        raise CoreError("run probability distribution is empty")
    return {key: value / total for key, value in result.items()}


def _tournament_distribution(keys: list[str], runs: list[dict[str, Any]]) -> dict[str, float]:
    positions = {key: index for index, key in enumerate(keys)}
    directional: dict[tuple[str, str], dict[str, float]] = {}
    for run in runs:
        pair = run.get("pair")
        probabilities = run.get("probabilities")
        if (not isinstance(pair, list) or len(pair) != 2 or pair[0] == pair[1]
                or any(key not in positions for key in pair) or not isinstance(probabilities, dict)):
            raise CoreError("malformed round-robin run")
        direction = (pair[0], pair[1])
        if direction in directional:
            raise CoreError("duplicate round-robin direction")
        vector = {key: float(probabilities.get(key, 0.0)) for key in pair}
        total = math.fsum(vector.values())
        if not math.isfinite(total) or total <= 0:
            raise CoreError("round-robin run probability distribution is empty")
        directional[direction] = {key: value / total for key, value in vector.items()}

    points = {key: 0.0 for key in keys}
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1:]:
            forward = directional.get((left, right))
            reverse = directional.get((right, left))
            if forward is None or reverse is None:
                raise CoreError("round-robin pair is incomplete")
            points[left] += (forward[left] + reverse[left]) / 2
            points[right] += (forward[right] + reverse[right]) / 2
    expected_count = len(keys) * (len(keys) - 1)
    if len(directional) != expected_count:
        raise CoreError("round-robin run set contains unexpected pairs")
    total = math.fsum(points.values())
    return {key: value / total for key, value in points.items()}


def aggregate(
    question: dict[str, Any],
    runs: list[dict[str, Any]],
    double_round_robin: bool,
) -> dict[str, Any]:
    """Convert parsed runs to the official Jev answer shape."""
    keys, descriptions = _question_parts(question)
    use_tournament = double_round_robin and question["type"] in {"choice", "score"} and len(keys) >= 2
    probabilities = (
        _tournament_distribution(keys, runs)
        if use_tournament
        else _single_distribution(keys, runs)
    )
    kind = question["type"]
    if kind == "noul":
        return {"type": "noul", "noul": probabilities["true"]}

    confidence = _confidence(probabilities.values())
    if kind == "choice":
        winner = max(keys, key=probabilities.__getitem__)
        return {
            "type": "choice",
            "choice": winner,
            "probabilities": probabilities,
            "confidence": confidence,
        }

    legend = dict(zip(keys, descriptions, strict=True))
    score = math.fsum(int(key) * probabilities[key] for key in keys)
    return {
        "type": "score",
        "score": score,
        "legend": legend,
        "probabilities": probabilities,
        "confidence": confidence,
    }
