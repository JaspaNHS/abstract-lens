---
title: Abstract Lens
emoji: 🔬
colorFrom: red
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Abstract Lens

Citation-grounded conversational assistant over the 8,255 abstracts of the 67th ASH
Annual Meeting (Blood 2025;146 Suppl 1). Answers are drawn only from the abstracts, with
a citation on every claim, and the assistant declines when the corpus does not contain
the answer.

Access is password-gated (set `APP_PASSWORD` as a Space secret). Synthesis requires an
`ANTHROPIC_API_KEY` Space secret.

Source code: https://github.com/JaspaNHS/abstract-lens
