"""Check Markdown files for math that GitHub's renderer does not display.

    python scripts/synth/lint_markdown_math.py doc/synthesizer.md

Flags: $$ blocks (use ```math), \\operatorname / \\boxed (not allowed),
backslash + punctuation inside inline $...$ math (eaten by Markdown, e.g.
"\\,"; write such expressions as $`...`$),
"|" inside inline math in table rows (splits the cell), matrices or line
breaks in inline math.
"""
import re
import sys

BAD_MACROS = ["\\operatorname", "\\boxed", "\\newcommand", "\\def"]
problems = 0
for path in sys.argv[1:]:
    text = open(path).read()
    in_fence = False
    for no, line in enumerate(text.splitlines(), 1):
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            if any(m in line for m in BAD_MACROS):
                print(f"{path}:{no}: macro not allowed on GitHub: {line.strip()}")
                problems += 1
            continue
        if "$$" in line:
            print(f"{path}:{no}: use a ```math block instead of $$")
            problems += 1
        code_free = re.sub(r"\$`.*?`\$", "", line)          # $`...`$ is safe
        code_free = re.sub(r"`[^`]*`", "", code_free)
        for m in re.finditer(r"(?<![\\$])\$(?!\$)(.+?)(?<!\\)\$", code_free):
            expr = m.group(1)
            issues = []
            if re.search(r"\\[,;:!{}\\|#&%_ ]", expr):
                issues.append("backslash + punctuation")
            if any(mac in expr for mac in BAD_MACROS):
                issues.append("macro not allowed")
            if "pmatrix" in expr or "\\\\" in expr:
                issues.append("matrix / line break inline")
            if line.lstrip().startswith("|") and "|" in expr:
                issues.append("'|' inside a table cell")
            if issues:
                print(f"{path}:{no}: {', '.join(issues)}: ${expr}$")
                problems += 1
print(f"{problems} problem(s)")
sys.exit(1 if problems else 0)
