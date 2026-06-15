import re

def classifierLLMParser(llm_output:str, output_mode:str)->dict|str:
    
    if not llm_output or not isinstance(llm_output, str):
        if output_mode == 'label_only':
            return float("nan")
        return {"label": float("nan"), "reason": None}
    
    text =llm_output.strip()
    label_pattern = re.compile(
        r"Label\s*:\s*\[?\s*(simple|complex)\s*\]?",
        re.IGNORECASE
    )

    label_matches = label_pattern.findall(text)
    label = str(label_matches[-1]).lower() if label_matches else None

    if output_mode == 'label_only':
        return label
    
    if output_mode == 'label_and_reasoning':

        reason = None

        reason_match = re.search(
            r"Reason\s*:\s*(.+)",
            text,
            re.IGNORECASE | re.DOTALL
        )

        if reason_match:
            reason = reason_match.group(1).strip()
        else:
            cleaned = label_pattern.sub("", text).strip()
            reason = cleaned if cleaned else None
        
        return {
            'label': label,
            'reason': reason
        }

def raterLLMParser(llm_output:str, output_mode:str)->dict|float:
    if not llm_output or not isinstance(llm_output, str):
        if output_mode == "rating_only":
            return float("nan")
        return {"rating": float("nan"), "reason": None}

    text = llm_output.strip()
    rating_pattern = re.compile(
        r"Rating\s*:\s*\[?\s*(0(?:\.5)?|1)\s*\]?",
        re.IGNORECASE
    )

    rating_matches = rating_pattern.findall(text)
    rating = float(rating_matches[-1]) if rating_matches else float("nan")

    if output_mode == "rating_only":
        return rating

    if output_mode == "rating_and_reasoning":
        reason = None

        reason_match = re.search(
            r"Reason\s*:\s*(.+)",
            text,
            re.IGNORECASE | re.DOTALL
        )

        if reason_match:
            reason = reason_match.group(1).strip()
        else:
            cleaned = rating_pattern.sub("", text).strip()
            reason = cleaned if cleaned else None

        return {
            "rating": rating,
            "reason": reason
        }

    raise ValueError(f"Unknown output_mode: {output_mode}")


def enhancerLLMParser(llm_output:str, output_mode:str)->str:
    pass

def decomposerLLMParser(llm_output:str, output_mode:str)->list[str]:

    pattern = r'^\d+\.\s+(.*)$'
    matches = re.findall(pattern, llm_output, re.MULTILINE)
    return matches

def parse_output(llm_output:str, parse_type:str, output_mode:str)->dict:
    if parse_type == "rater":
        return raterLLMParser(llm_output, output_mode)
    elif parse_type == "classifier":
        return classifierLLMParser(llm_output, output_mode)
    elif parse_type == "enhancer":
        return enhancerLLMParser(llm_output, output_mode)
    elif parse_type == "decomposer":
        return decomposerLLMParser(llm_output, output_mode)
    else:
        raise(f"Parser not defined for {parse_type}")
