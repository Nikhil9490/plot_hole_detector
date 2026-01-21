# plot_hole_detector

<img width="2389" height="1531" alt="image" src="https://github.com/user-attachments/assets/fd10b7a5-f774-481a-9b0c-6697a4bcd370" />

This project started as an attempt to answer a very specific question.

Can we detect genuine plot holes while someone is writing, without turning every stylistic choice or ambiguity into a false alarm?

Most existing tools, including general purpose AI assistants, are good at giving feedback when asked. They are not good at quietly verifying logical consistency in the background. They tend to over-flag, hallucinate issues, or confuse subjective narration with factual contradiction. This project explores how to make that process more conservative, explainable, and reliable.

The goal was never to replace human judgment or compete with large language models at creativity. The goal was to treat plot holes as a verification problem rather than a suggestion problem.

The system analyzes narrative text and detects only hard logical or continuity contradictions. These include explicit age mismatches, time or timeline inconsistencies, mutually exclusive states, and factual statements that negate earlier facts.

It deliberately ignores stylistic issues, narrative voice, unreliable narrators, internal monologue, hyperbole, and missing details that do not form a direct contradiction. The emphasis is on high confidence findings with clear textual evidence.

The tool is designed to work incrementally, making it suitable for background analysis while writing rather than post-hoc critique.

The backend is built with FastAPI and exposes a simple analysis endpoint. Text is sent to the backend, where two complementary approaches are applied.

First, a large language model is used as a reasoning engine. The model is prompted to act conservatively and to report only explicit contradictions. It is constrained to return structured JSON with quoted evidence taken directly from the input text.

Second, symbolic validation is applied for cases where deterministic logic is more reliable than probabilistic reasoning. A concrete example is age consistency. If a character is described as twelve years old and also celebrating a twenty second birthday, that contradiction is detected deterministically even if the language model misses it or phrases the evidence imperfectly.

The final output is a filtered set of issues that pass both reasoning and validation checks. If no high confidence issues are found, the system returns an empty result rather than speculative feedback.

General purpose AI tools for writing are interactive and suggestion-oriented. They work when the user asks a question. This project explores a different space.

This tool is passive. It is meant to run quietly in the background, analyze small units of text as they are written, and surface only hard contradictions. It prioritizes trust over coverage and silence over verbosity.

Rather than trying to be smarter than a language model, the system is designed to be stricter than one. The language model generates candidate reasoning. The system decides what is safe enough to show.

This project demonstrates how to combine LLM reasoning with deterministic validation to improve trustworthiness. It shows how to design prompts for conservative behavior, how to post-process model output to avoid hallucinations, and how to structure AI systems around verification rather than creativity.

It also highlights the limits of purely rule-based approaches and the necessity of language models for narrative understanding, while still showing where symbolic logic remains superior.

This project is best understood as static analysis for fiction. It applies ideas from program verification to narrative text and treats plot holes as logical contradictions rather than stylistic flaws.

Even where the project ends, the lessons from it carry forward into any system that depends on trustworthy AI reasoning.
