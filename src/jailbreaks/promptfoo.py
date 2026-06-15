from urllib.parse import quote

def url_encoding(prompt: str) -> str:
    text = quote(prompt)
    return text


def camelcase_encode(prompt: str) -> str:
    words = prompt.split()
    return words[0].lower() + "".join(w.capitalize() for w in words[1:])


def homoglyph_encode(prompt: str) -> str:
    mapping = {
        "a": "а",  # Cyrillic
        "e": "е",  # Cyrillic
        "i": "і",  # Ukrainian
        "o": "ο",  # Greek
        "c": "с",  # Cyrillic
        "p": "р",  # Cyrillic
        "y": "у",  # Cyrillic
        "x": "х",  # Cyrillic
        "h": "һ",  # Cyrillic
    }
    return "".join(mapping.get(ch.lower(), ch) for ch in prompt)


def hex_encode(prompt: str) -> str:
    return prompt.encode("utf-8").hex()


def morse_code_encode(prompt: str) -> str:
    morse_dict = {
        "a": ".-", "b": "-...", "c": "-.-.", "d": "-..", "e": ".", "f": "..-.",
        "g": "--.", "h": "....", "i": "..", "j": ".---", "k": "-.-", "l": ".-..",
        "m": "--", "n": "-.", "o": "---", "p": ".--.", "q": "--.-", "r": ".-.",
        "s": "...", "t": "-", "u": "..-", "v": "...-", "w": ".--", "x": "-..-",
        "y": "-.--", "z": "--..",
        "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
        "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
        '.': '.-.-.-',
        ',': '--..--',
        '?': '..--..',
        "'": '.----.',
        '!': '-.-.--',
        '/': '-..-.',
        '(': '-.--.',
        ')': '-.--.-',
        '&': '.-...',
        ':': '---...',
        ';': '-.-.-.',
        '=': '-...-',
        '+': '.-.-.',
        '-': '-....-',
        '_': '..--.-',
        '"': '.-..-.',
        '$': '...-..-',
        '@': '.--.-.',
    }
    return " ".join(morse_dict.get(ch.lower(), ch) for ch in prompt)


def atbash(prompt: str) -> str:
    return "".join(
        chr(ord('Z') - ord(c) + ord('A')) if 'A' <= c <= 'Z'
        else chr(ord('z') - ord(c) + ord('a')) if 'a' <= c <= 'z'
        else c
        for c in prompt
    )

def octal_encode(prompt: str) -> str:
    return " ".join(f"{byte:03o}" for byte in prompt.encode("utf-8"))



jailbreaks = [
    "url_encoding",
    "camelcase_encode",
    "homoglyph_encode",
    "hex_encode",
    "morse_code_encode",
    "atbash"
    "octal_encode"
]