# Video, slides and reflection

中文版：[video.zh-CN.md](video.zh-CN.md)

All times are Singapore time.

## What we hand in

1. **Video.** At most 10 minutes. We aim for 9:40.
2. **Code.** `.ipynb` or `.py` files: the package, one driver notebook and the Streamlit app. Data and model weights are fetched by download scripts.
3. **Group reflection.** One document. Each member writes a few sentences on what they did. All six sign it.
4. **Peer evaluation.** It opens later on the course site. Everyone does it alone.

One submission for the whole team. We can submit as often as we like, and only the last attempt is graded. So any attempt after Sun 15 Nov 23:59 makes the whole submission late. Only the lead submits.

## Rules for the video

- It shows Python code running, slides for the theory, and references.
- Our own voices only. No generated voice.
- Python only. No MATLAB or other languages.
- Every member presents their own work. Faces are optional; voice is enough.

## Parts

| Time | Speaker | Content |
|---|---|---|
| 0:00–1:00 | A · Xingyang | The teacher's problem, the "target level" wording, the research questions, our specifications |
| 1:00–2:20 | B · Wenyue | The datasets and where the level labels come from, the data checks, the published baselines, what the audit of our worst errors found |
| 2:20–3:40 | D · Hailing | Floors, classical and feature models with a cross-validation run on screen, global SHAP, the open LLM without training |
| 3:40–5:30 | E | BiLSTM and the CORN head with a training cell running, transformers, Optuna, stable training on ARC, our numbers next to the published ones |
| 5:30–6:30 | C · Qiutian | How one run starts from one config, the queue of runs on the GPU, GPU-hours, the efficiency and latency table |
| 6:30–8:50 | F | Cue words from LIME, transfer and extrapolation, then the live demo: score a question, edit it and score it again |
| 8:50–9:40 | A · Xingyang | Pass or fail for each specification, limitations, future work, references |

If the video is too long, we cut content. We do not cut a speaker.

## Slides

- B shares the template on Fri 13 Nov at 12:00. Everyone uses it.
- Slide drafts: Sat 14 Nov 10:00. Numbers may still be placeholders.
- Final slides: Sat 14 Nov 16:00. B proofreads them by 18:00.
- Say "target level". Never claim difficulty measured on students.
- Every number on a slide comes from a table built from `results/runs.jsonl`.

## Recording

- Timed dry run online: Sat 14 Nov 16:30.
- Record your own part, voice and screen, after B's proofreading. Send it to the lead by Sat 14 Nov 23:59.
- Show your code running or the demo for at least part of your time. Screenshots of code are not enough.
- Record at 1920 × 1080 in a quiet room. Stay inside your time.
- Speak in your own words. Small slips are fine.
- Name the file `part<number>_<share>.mp4`, for example `part3_D.mp4`. Recordings go to the shared drive, not to this repository.
- The lead joins the parts by Sun 15 Nov 11:00. Everyone watches the full video at 12:00 and signs off by 14:00.

## Reflection document

Each member writes their own lines in their own words and sends them to B by Sun 15 Nov 10:00. B builds one document. Everyone signs it by 14:00.

The filled document has names, matriculation numbers and signatures. **Keep it out of this public repository.**

```text
Group Capstone: Group Reflection (Table 38)
Project: LevelLens - estimating the target difficulty level of exam questions

1. Name: ____________   Matric No.: ____________   Share: ______
   What I did (3-5 sentences):
   What I learned:
   Signature: ____________   Date: ____________

(repeat for all six members)

Team reflection
- What worked well:
- What we would do differently:
- Acknowledged and signed by all members above.
```

## Peer evaluation

- Everyone rates all five teammates and writes a comment for each.
- Skipping it, or doing it halfway, costs 5% of your own project mark.
- Make your work visible: commit often, post results in the group chat, and join the dry run and the reviews.
