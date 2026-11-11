from pathlib import Path

import fitz


output = Path("imports/demo")
output.mkdir(parents=True, exist_ok=True)


def write_pdf(filename, title, paragraphs):
    destination = output / filename

    if destination.exists():
        print("Already exists:", destination)
        return

    document = fitz.open()

    try:
        page = document.new_page()
        text = title + "\n\n" + "\n\n".join(paragraphs)

        remaining = page.insert_textbox(
            fitz.Rect(50, 50, 550, 780),
            text,
            fontsize=11,
        )

        if remaining < 0:
            raise RuntimeError(
                f"Text overflow while generating {filename}."
            )

        document.set_metadata({"title": title})
        document.save(destination)

    finally:
        document.close()

    print("Created:", destination)


write_pdf(
    "technical.pdf",
    "Technical example: quadratic energy",
    [
        "1. Assumptions",
        "Let x be a real scalar and let k be a positive stiffness constant.",
        "2. Energy definition",
        "Equation (2.1): E = 0.5 * k * x^2.",
        "The variable x denotes displacement. "
        "The constant k denotes stiffness.",
        "Differentiating Equation (2.1) with respect to x gives "
        "dE/dx = k*x.",
    ],
)

write_pdf(
    "legal.pdf",
    "Fictional local policy - NOT real law",
    [
        "Section 4.2 - Records retention",
        "This fictional demonstration policy states that project "
        "records must be retained for seven years after project completion.",
        "Section 4.3 - Exceptions",
        "An active litigation hold suspends routine disposal.",
        "This demonstration is not a statement of current law.",
    ],
)

tex_file = output / "technical.tex"

if not tex_file.exists():
    tex_file.write_text(
        r"""\documentclass{article}
\title{Quadratic energy source}
\begin{document}
\section{Assumptions}
Let \(x\) be displacement and \(k>0\) be stiffness.
\section{Energy}
\begin{equation}\label{eq:energy}
E=\frac12 kx^2
\end{equation}
Equation \ref{eq:energy} yields \(dE/dx=kx\).
\end{document}
""",
        encoding="utf-8",
    )
    print("Created:", tex_file)
else:
    print("Already exists:", tex_file)
