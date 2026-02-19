"""
🥗 Product Ingredient Health Analyzer Agent
=============================================
Analyzes product ingredient images using Google Gemini Vision
and provides a detailed health score with explanation.

"""

import os
import sys
import base64
import argparse
import json
import re
from pathlib import Path
from PIL import Image
import io
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
# ──────────────────────────────────────────────────────────────
# ANSI Colors for beautiful terminal output
# ──────────────────────────────────────────────────────────────
class Colors:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    BG_RED  = "\033[41m"
    BG_GREEN= "\033[42m"
    BG_YELLOW = "\033[43m"

C = Colors


# ──────────────────────────────────────────────────────────────
# Initialize Gemini Vision LLM
# ──────────────────────────────────────────────────────────────
def get_llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print(f"\n{C.RED}{C.BOLD}❌  GOOGLE_API_KEY not set!{C.RESET}")
        print(f"{C.YELLOW}Get your free key at: https://aistudio.google.com/app/apikey{C.RESET}")
        print(f"{C.CYAN}Then run: export GOOGLE_API_KEY='your-key-here'{C.RESET}\n")
        sys.exit(1)

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=api_key,
        temperature=0.2,          # low temp → consistent analysis
        convert_system_message_to_human=True,
    )


# ──────────────────────────────────────────────────────────────
# Image Utilities
# ──────────────────────────────────────────────────────────────
def load_image_as_base64(image_path: str) -> tuple[str, str]:
    """Load an image and return (base64_string, mime_type)."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    ext = path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    mime_type = mime_map.get(ext, "image/jpeg")

    # Resize if too large (keep under 4MB)
    with Image.open(image_path) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        max_size = (1600, 1600)
        img.thumbnail(max_size, Image.LANCZOS)

        buffer = io.BytesIO()
        save_format = "JPEG" if mime_type == "image/jpeg" else "PNG"
        img.save(buffer, format=save_format, quality=90)
        image_bytes = buffer.getvalue()

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return b64, mime_type


# ──────────────────────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an expert nutritionist, food scientist, and health analyst AI.
Your job is to analyze product ingredient lists from images and evaluate their health impact.

When analyzing ingredients, consider:
- Presence of artificial additives, preservatives, colors, and flavors
- Sugar content and types (natural vs. added sugars, HFCS, etc.)
- Unhealthy fats (trans fats, hydrogenated oils, excessive saturated fats)
- Sodium/salt levels
- Allergens and common irritants
- Beneficial nutrients, vitamins, minerals, and whole food ingredients
- Overall ingredient quality and processing level (NOVA classification)
- Known harmful ingredients (e.g., BHA, BHT, carrageenan, Red 40, MSG, etc.)
- Organic vs. conventional ingredients when mentioned

Your response MUST be a valid JSON object only (no markdown, no extra text) with this exact structure:
{
  "product_detected": "name or description of product if visible",
  "ingredients_found": ["list", "of", "all", "detected", "ingredients"],
  "health_score": <integer 0-100>,
  "verdict": "EXCELLENT | GOOD | MODERATE | POOR | VERY POOR",
  "summary": "2-3 sentence overall summary",
  "positive_aspects": ["list of healthy/good ingredients or attributes"],
  "concerns": ["list of concerning ingredients or health issues"],
  "harmful_ingredients": ["specifically dangerous or very unhealthy ingredients if any"],
  "recommended_for": ["groups of people this product is suitable for"],
  "avoid_if": ["conditions or groups who should avoid this product"],
  "healthier_alternatives": ["2-3 suggestions for healthier alternatives"],
  "detailed_analysis": "paragraph with in-depth nutritional and ingredient analysis"
}

Health Score Guide:
  90-100 → EXCELLENT  (whole foods, no harmful additives, highly nutritious)
  70-89  → GOOD       (mostly healthy, minor concerns)
  50-69  → MODERATE   (mixed, some concerns, occasional consumption okay)
  30-49  → POOR       (several harmful ingredients, limit consumption)
  0-29   → VERY POOR  (many harmful additives, highly processed, avoid)
"""


# ──────────────────────────────────────────────────────────────
# Core Agent: Analyze Ingredients
# ──────────────────────────────────────────────────────────────
def analyze_ingredients(image_path: str, llm: ChatGoogleGenerativeAI) -> dict:
    """Send image to Gemini Vision and get health analysis."""
    print(f"\n{C.CYAN}🔍  Loading image...{C.RESET}")
    b64_image, mime_type = load_image_as_base64(image_path)

    print(f"{C.CYAN}🧠  Analyzing ingredients with Gemini Vision...{C.RESET}\n")

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Please analyze the ingredient list visible in this image. "
                    "Extract ALL ingredients you can read and provide a comprehensive health analysis. "
                    "Return ONLY a valid JSON object as described in your instructions."
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{b64_image}"
                },
            },
        ]
    )

    system_msg = SystemMessage(content=SYSTEM_PROMPT)
    response = llm.invoke([system_msg, message])

    # Parse JSON from response
    raw = response.content.strip()
    # Remove markdown code blocks if present
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    raw = raw.rstrip("`").strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from the text
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            raise ValueError(f"Could not parse JSON response:\n{raw}")

    return result


# ──────────────────────────────────────────────────────────────
# Display Functions
# ──────────────────────────────────────────────────────────────
def get_score_color(score: int) -> str:
    if score >= 90: return C.GREEN
    if score >= 70: return C.CYAN
    if score >= 50: return C.YELLOW
    if score >= 30: return C.RED
    return C.BG_RED + C.WHITE


def get_score_bar(score: int, width: int = 40) -> str:
    filled = int((score / 100) * width)
    color = get_score_color(score)
    bar = "█" * filled + "░" * (width - filled)
    return f"{color}{bar}{C.RESET}"


def get_verdict_badge(verdict: str) -> str:
    badges = {
        "EXCELLENT": f"{C.BG_GREEN}{C.WHITE}{C.BOLD} ✅ EXCELLENT {C.RESET}",
        "GOOD":      f"{C.GREEN}{C.BOLD} ✅ GOOD {C.RESET}",
        "MODERATE":  f"{C.YELLOW}{C.BOLD} ⚠️  MODERATE {C.RESET}",
        "POOR":      f"{C.RED}{C.BOLD} ❌ POOR {C.RESET}",
        "VERY POOR": f"{C.BG_RED}{C.WHITE}{C.BOLD} 🚫 VERY POOR {C.RESET}",
    }
    return badges.get(verdict.upper(), f"{C.MAGENTA}{verdict}{C.RESET}")


def print_section(title: str, color: str = C.CYAN):
    print(f"\n{color}{C.BOLD}{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}{C.RESET}")


def display_results(data: dict):
    """Pretty-print the health analysis results."""
    score    = data.get("health_score", 0)
    verdict  = data.get("verdict", "UNKNOWN")
    product  = data.get("product_detected", "Unknown Product")

    # ── Header ──
    print(f"\n{C.BOLD}{C.MAGENTA}{'═'*55}")
    print(f"  🥗  INGREDIENT HEALTH ANALYSIS REPORT")
    print(f"{'═'*55}{C.RESET}")

    # ── Product ──
    print(f"\n{C.BOLD}📦 Product:{C.RESET} {C.WHITE}{product}{C.RESET}")

    # ── Score ──
    print_section("🎯 HEALTH SCORE", get_score_color(score))
    score_color = get_score_color(score)
    print(f"  {get_score_bar(score)}")
    print(f"  {score_color}{C.BOLD}  {score}/100  {C.RESET}  {get_verdict_badge(verdict)}")

    # ── Summary ──
    print_section("📋 SUMMARY", C.BLUE)
    summary = data.get("summary", "")
    for line in _wrap_text(summary, 53):
        print(f"  {line}")

    # ── Ingredients Found ──
    ingredients = data.get("ingredients_found", [])
    if ingredients:
        print_section("🧪 INGREDIENTS DETECTED", C.WHITE)
        ing_line = ", ".join(ingredients)
        for line in _wrap_text(ing_line, 53):
            print(f"  {C.WHITE}{line}{C.RESET}")

    # ── Positives ──
    positives = data.get("positive_aspects", [])
    if positives:
        print_section("✅ POSITIVE ASPECTS", C.GREEN)
        for item in positives:
            print(f"  {C.GREEN}+{C.RESET} {item}")

    # ── Concerns ──
    concerns = data.get("concerns", [])
    if concerns:
        print_section("⚠️  CONCERNS", C.YELLOW)
        for item in concerns:
            print(f"  {C.YELLOW}!{C.RESET} {item}")

    # ── Harmful Ingredients ──
    harmful = data.get("harmful_ingredients", [])
    if harmful:
        print_section("🚫 HARMFUL INGREDIENTS", C.RED)
        for item in harmful:
            print(f"  {C.RED}✗{C.RESET} {item}")

    # ── Who Should Avoid ──
    avoid = data.get("avoid_if", [])
    if avoid:
        print_section("🚷 AVOID IF YOU HAVE", C.RED)
        for item in avoid:
            print(f"  {C.RED}→{C.RESET} {item}")

    # ── Recommended For ──
    rec = data.get("recommended_for", [])
    if rec:
        print_section("👍 SUITABLE FOR", C.GREEN)
        for item in rec:
            print(f"  {C.GREEN}→{C.RESET} {item}")

    # ── Healthier Alternatives ──
    alts = data.get("healthier_alternatives", [])
    if alts:
        print_section("💡 HEALTHIER ALTERNATIVES", C.CYAN)
        for item in alts:
            print(f"  {C.CYAN}→{C.RESET} {item}")

    # ── Detailed Analysis ──
    detailed = data.get("detailed_analysis", "")
    if detailed:
        print_section("🔬 DETAILED ANALYSIS", C.MAGENTA)
        for line in _wrap_text(detailed, 53):
            print(f"  {line}")

    # ── Footer ──
    print(f"\n{C.BOLD}{C.MAGENTA}{'═'*55}{C.RESET}\n")

    # ── Final Verdict ──
    if score >= 70:
        print(f"  {C.GREEN}{C.BOLD}🟢 OVERALL: This product is generally HEALTHY to consume.{C.RESET}")
    elif score >= 50:
        print(f"  {C.YELLOW}{C.BOLD}🟡 OVERALL: Consume this product in MODERATION.{C.RESET}")
    else:
        print(f"  {C.RED}{C.BOLD}🔴 OVERALL: This product is NOT recommended for regular consumption.{C.RESET}")

    print(f"\n{C.BOLD}{C.MAGENTA}{'═'*55}{C.RESET}\n")


def _wrap_text(text: str, width: int) -> list[str]:
    """Simple word wrapper."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 <= width:
            current = (current + " " + word).strip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


# ──────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="🥗 Ingredient Health Analyzer — powered by Google Gemini Vision"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Path to the product ingredient image",
    )
    args = parser.parse_args()

    print(f"{C.BOLD}{C.MAGENTA}")
    print("╔══════════════════════════════════════════════════════╗")
    print("║   🥗  Product Ingredient Health Analyzer Agent       ║")
    print("║       Powered by Google Gemini Vision + LangChain   ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(C.RESET)

    llm = get_llm()

    # Interactive loop
    while True:
        if args.image:
            image_path = args.image
            args.image = None  # only use CLI arg once
        else:
            print(f"{C.CYAN}📁 Enter image path (or 'quit' to exit):{C.RESET}")
            image_path = input(f"{C.WHITE}   Path: {C.RESET}").strip().strip('"').strip("'")

        if image_path.lower() in ("quit", "exit", "q", ""):
            print(f"\n{C.CYAN}Goodbye! Stay healthy! 🥦{C.RESET}\n")
            break

        try:
            result = analyze_ingredients(image_path, llm)
            display_results(result)

            # Save JSON report
            output_file = Path(image_path).stem + "_health_report.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"{C.CYAN}💾 Full report saved to: {output_file}{C.RESET}\n")

        except FileNotFoundError as e:
            print(f"\n{C.RED}❌ File Error: {e}{C.RESET}\n")
        except json.JSONDecodeError as e:
            print(f"\n{C.RED}❌ Could not parse AI response: {e}{C.RESET}\n")
        except Exception as e:
            print(f"\n{C.RED}❌ Unexpected error: {e}{C.RESET}\n")

        # Ask to analyze another
        again = input(f"{C.YELLOW}Analyze another product? (y/n): {C.RESET}").strip().lower()
        if again != "y":
            print(f"\n{C.CYAN}Goodbye! Stay healthy! 🥦{C.RESET}\n")
            break


if __name__ == "__main__":
    main()