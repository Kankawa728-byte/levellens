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
| 3:40–5:30 | E · Jikai | BiLSTM and the CORN head with a training cell running, transformers, Optuna, stable training on ARC, our numbers next to the published ones |
| 5:30–6:30 | C · Qiutian | How one run starts from one config, the queue of runs on the GPU, GPU-hours, the efficiency and latency table |
| 6:30–8:50 | F · Zhifei | Cue words from LIME, transfer and extrapolation, then the live demo: score a question, edit it and score it again |
| 8:50–9:40 | A · Xingyang | Pass or fail for each specification, limitations, future work, references |

If the video is too long, we cut content. We do not cut a speaker.

## Slides

- B shares the template on Fri 13 Nov at 12:00. Everyone uses it.
- Slide drafts: Sat 14 Nov 10:00. Numbers may still be placeholders.
- Final slides: Sat 14 Nov 16:00. B proofreads them by 18:00.
- Say "target level". Never claim difficulty measured on students.
- Every number on a slide comes from a table built from `results/runs.jsonl`.

How to use the template:

- The template is one deck in storyboard order. Open it in the desktop version of PowerPoint. The fonts are inside the file, so you install nothing.
- Find the section of your part. Replace the draft text on your slides. Delete the red Draft tag when a slide is final.
- The storyboard is in the deck. One hidden slide lists every slide with its speaker and its seconds. The notes of each slide give its seconds and the points to make.
- One message for each slide. The title states the message, in sentence case, with no full stop.
- Figures have a white background, 200 dpi or more, and no title inside the figure.

## Recording

- Timed dry run online: Sat 14 Nov 16:30.
- Record your own part, voice and screen, after B's proofreading. Send it to the lead by Sat 14 Nov 23:59.
- Show your code running or the demo for at least part of your time. Screenshots of code are not enough.
- Record at 1920 × 1080 in a quiet room. Stay inside your time.
- Speak in your own words. Small slips are fine.
- Name the file `part<number>_<share>.mp4`, for example `part3_D.mp4`. Recordings go to the shared drive, not to this repository.
- The lead joins the parts by Sun 15 Nov 11:00. Everyone watches the full video at 12:00 and signs off by 14:00.

### Before you record

1. Make sure that your slides are proofread and that no slide has a Draft tag.
2. Close chat, mail and every program that shows notifications. Turn on "Do not disturb".
3. Hide what is private: matriculation number, email, access tokens, the bookmarks bar, and folder names that show your full name.
4. Make the text large: 16 pt or more in the terminal and the editor, or a zoom of 150% in the notebook and the browser.
5. For a part with code, set `LEVELLENS_RESULTS` to a scratch folder first. Then the run that you record adds no row to `results/runs.jsonl`.
6. Run the code once before you record. The first run downloads the models and is slow. Go through your slides once in the slide show, with every click.
7. Record ten seconds as a test and listen to them. A headset microphone is better than the microphone of a laptop.

### Settings

- 1920 × 1080, 30 frames per second, MP4 with H.264, sound at 48 kHz.
- Use any program that records the screen and the microphone, for example OBS Studio or the local recording of a meeting program.
- One file for each part. The lead has two files, `part1_A.mp4` and `part7_A.mp4`.

### While you record

1. Leave one second of silence at the start and one at the end. The lead needs them to join the parts.
2. Say your name and what you did in your first sentence.
3. Follow the seconds in the notes of each slide.
4. Some slides show their content in steps. The notes of the slide say how many clicks it needs and what each click shows. Stop talking for half a second while a slide changes.
5. For a slide marked "Live", show the slide for about two seconds. Then switch to the window in which the code or the app runs.
6. After a slip, stop, wait one second and say the sentence again. The lead cuts the first attempt.

### After you record

1. Watch the file once. Check the length, the sound, and that nothing private is on the screen.
2. Name the file and put it on the shared drive.
3. Comment `done recording` on the issue of your share.

## Recording schedule

All recordings are due on Sat 14 Nov at 23:59. A part that needs nothing first can be recorded at once.

| File | Speaker | Length | Live part | Needs first |
|---|---|---|---|---|
| `part1_A.mp4` | A · Xingyang | 1:00 | none | nothing |
| `part2_B.mp4` | B · Wenyue | 1:20 | none | the audit counts, from the final models |
| `part3_D.mp4` | D · Hailing | 1:20 | a five-fold search | nothing |
| `part4_E.mp4` | E · Jikai | 1:50 | a training cell | the final transformer rows |
| `part5_C.mp4` | C · Qiutian | 1:00 | the queue of runs on the GPU | the efficiency table |
| `part6_F.mp4` | F · Zhifei | 2:20 | the Streamlit app | the final models and the final figures |
| `part7_A.mp4` | A · Xingyang | 0:50 | none | pass or fail for each specification |

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
