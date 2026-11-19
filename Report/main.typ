#set document(
  title: "Reproducing DBSCAN: A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise",
  author: ("Phùng Bảo Khang", "Nguyễn Minh Đức", "Nguyễn Phạm Tuấn Đạt", "Nguyễn Hoàng Bảo Long", "Văn Phú Đức"),
)

#set page(
  paper: "a4",
  margin: (x: 2.2cm, y: 2.2cm),
  numbering: "1",
)

#set text(
  font: ("Liberation Serif", "New Computer Modern", "DejaVu Serif"),
  size: 10.5pt,
  lang: "en",
)

#set heading(numbering: "1.1")
#set par(justify: true, leading: 0.65em)

// Title Block
#align(center)[
  #text(size: 11pt, weight: "semibold")[
    University of Science, VNU-HCM — Faculty of Information Technology \
    CSC14005: Introduction to Machine Learning
  ]
  #v(1.2em)
  #text(size: 18pt, weight: "bold")[
    Paper Reproduction Report: DBSCAN (KDD-96)
  ]
  #v(0.5em)
  #text(size: 11pt, style: "italic")[
    A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise
  ]
  #v(1.2em)
  #table(
    columns: (auto, auto),
    stroke: none,
    align: (left, left),
    [*Student IDs & Names:*], [24127052 – Phùng Bảo Khang (M1, Lead) \ 24127345 – Nguyễn Minh Đức (M2) \ 24127342 – Nguyễn Phạm Tuấn Đạt (M3) \ 24127201 – Nguyễn Hoàng Bảo Long (M4) \ 24127346 – Văn Phú Đức (M5)],
    [*Repository:*], [https://github.com/AndyKhang404/IntroML-DBSCAN],
    [*Video URL:*], [See `url.txt` (pending upload)],
  )
]

#v(1em)
#outline(indent: auto)
#pagebreak()

#include "sections/01_theory.typ"
#include "sections/02_reproduction.typ"
#include "sections/03_bonus.typ"
#include "sections/04_team.typ"
#include "sections/05_ai_use.typ"

#bibliography("refs.bib", style: "ieee")
