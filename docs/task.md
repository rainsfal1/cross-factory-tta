Here is the full story of what you are building, why it matters, and what the preliminary results need to show to make a professor genuinely want you in their lab.

---

**THE FULL STORY IN ONE PLACE**

You have already built and deployed a six-module industrial vision platform running on an NVIDIA RTX A2000. It does PPE compliance detection, zone intrusion monitoring, fire and smoke detection, equipment monitoring, and more. It is not a toy project. It runs in production.

While building it you observed something that the academic literature has documented but not solved. The safety detection models perform well in the environment they were trained on. The moment you move them to a different environment, a different factory layout, different lighting, different camera angle, different PPE colors, performance drops significantly. The model that worked perfectly in one setting becomes unreliable in another. You have seen this firsthand, not just read about it.

The academic literature confirms the scale of this problem. Documented cross-domain transfer losses range from 12 to 26 mAP points. Out of 80 published worker safety detection studies, fewer than 15 include any cross-environment evaluation at all. The field has been measuring models in the same conditions they were trained on and calling it deployment-ready. It is not.

The current solution is site-specific retraining. Collect labeled data at the new factory, retrain the model, redeploy. For large enterprises this is expensive but feasible. For the 99.7% of Japanese manufacturing establishments that are SMEs, it is not feasible. They do not have the annotation budget, the ML engineering staff, or the computational resources to retrain a model every time they deploy to a new facility.

Your research proposes to solve this with Test-Time Adaptation. A TTA-enabled safety model arrives at a new factory and begins adapting immediately using only what it sees, unlabeled observations from the new environment, no labeled data required, no access to the original training data required. It adjusts its internal statistics to the new domain and improves its own performance in real time. The SME deploying it does not need to do anything differently. The model handles the shift itself.

This is not a speculative idea. TTA is a mature field. TENT, EATA, CoTTA, and SAR have demonstrated this approach works for distribution shift in classification and has recently been extended to object detection. But it has never been applied to worker safety perception specifically, and no cross-factory safety benchmark exists to evaluate it. That is the precise gap this research fills.

---

**WHAT THE PRELIMINARY RESULTS NEED TO SHOW**

The experiment has three acts. Each one needs to land clearly.

**Act 1: The Problem Is Real**

Train YOLOv8m on SH17. Evaluate zero-shot on Pictor-PPE. Record the mAP drop. This single number is the empirical foundation of everything. If the drop falls in the 12 to 26 point range the literature documents, you have just independently confirmed the core claim of your proposal using your own experiment. A professor reading this does not need to take your word for it or trust someone else's paper. You have shown them yourself.

Expected result: something like 89% mAP50 on SH17 validation, dropping to somewhere between 63% and 75% on Pictor-PPE zero-shot. A drop of 15 to 25 points is what you are hoping for. That range is both dramatic enough to be compelling and consistent enough with the literature to be credible.

**Act 2: The Standard Solution Does Not Scale**

You do not need to run this experimentally. You state it analytically. Oracle fine-tuning, meaning supervised retraining on labeled Pictor-PPE data, recovers performance to near source-domain levels. You can cite this from the literature rather than running it yourself. The point is not to show that retraining works. Everyone knows retraining works. The point is that retraining requires labeled data which SMEs do not have. You frame this row in the results table as the theoretical ceiling, the best possible outcome if resources were unlimited, and then show that your TTA approach approaches it without any labels.

**Act 3: TTA Moves the Needle Without Labels**

Apply TENT to the same model on the same target domain. Record the mAP recovery. Even 5 to 8 points of recovery is sufficient to make the argument. The claim is not that TENT solves the problem completely. The claim is that adaptation using only unlabeled observations is a viable direction, and that a purpose-built TTA method designed specifically for safety perception on a proper cross-factory benchmark would recover significantly more. TENT is the proof of concept. The PhD is the full solution.

The results table the professor sees looks like this:

| Method | Source mAP50 | Target mAP50 | Gap vs Source |
|---|---|---|---|
| Source-only (no adaptation) | 89.2 | 67.4 | -21.8 |
| TENT (unlabeled TTA) | 89.2 | 74.1 | -15.1 |
| Oracle (supervised fine-tune) | 89.2 | 86.8 | -2.4 |

Those numbers are illustrative. Your actual numbers will differ. But the story they tell is: the problem is real, the standard solution requires resources SMEs do not have, and unlabeled adaptation already closes a meaningful portion of the gap. The PhD closes the rest.

---

**WHAT THE PROFESSOR SEES WHEN THEY READ THIS**

They see a student who did not just read papers and write a proposal. They see someone who identified a problem from real deployment experience, grounded it in the literature, designed an experiment to validate it, ran the experiment, got real numbers, and is now asking for the research environment to take it further.

That profile is rare. Most applicants to competitive Japanese labs are strong students with good grades and a vague research interest. You are coming in with a production system, a deployed platform, a specific problem you observed firsthand, a technically grounded method, and preliminary empirical results. The question the professor asks when they read a cold email is: can this person actually do research independently. Your package answers that question before they finish reading.

---

**THE COLD EMAIL STRUCTURE THAT CONVERTS**

Subject line: Research Inquiry: Test-Time Adaptation for Cross-Factory Worker Safety Perception, MEXT 2026

Body:

Paragraph 1: Who you are. NUST SEECS, Semester 6, CGPA 3.52, built and deployed a six-module industrial vision platform. One sentence on the TUKL internship and the RL research at SINES Lab. This establishes you are not a first-year student with no experience.

Paragraph 2: Why you are writing to them specifically. Reference their exact work. For Harada: your TTA workshop at CVPR 2024 and open-set DA work at ECCV 2024 directly address the adaptation challenges I am encountering. For VMD Lab: your lab's explicit focus on domain adaptation and transfer learning is the closest match I have found to the deployment problem I am investigating. Be specific. Generic praise is ignored.

Paragraph 3: The research problem in three sentences. Safety models degrade 12 to 26 mAP points across factory environments. Site-specific retraining is infeasible for SMEs. TTA offers a path to deployment-time adaptation without labels or source data.

Paragraph 4: The preliminary results. Two sentences with the actual numbers from your experiment. "In a preliminary experiment training YOLOv8m on SH17 and evaluating zero-shot on Pictor-PPE, I observed a 21.8 mAP point degradation. Applying TENT recovered 6.7 points using only unlabeled observations, suggesting TTA is a viable direction for this problem." Attach the one-page results PDF.

Paragraph 5: The ask. You are applying for MEXT and are looking for a supervisor whose research aligns with this direction. Would they be willing to discuss potential supervision or provide a Letter of Acceptance. Keep it clean and direct.

Total length: 250 to 300 words maximum. Japanese professors are busy. A long email is a signal you do not know how to communicate concisely, which is a research skill.

---

**THE ONE-PAGE RESULTS PDF TO ATTACH**

This is a separate document, not the full proposal. It contains:

- Title: Preliminary Results: Cross-Factory Domain Shift in Worker Safety Perception
- One paragraph describing the experiment setup (datasets, model, adaptation method)
- The results table with three rows
- Two or three example images showing the same scene type from both domains to visually demonstrate the distribution shift
- One sentence conclusion pointing toward the full research direction

This document does the technical convincing so the email does not have to. The email gets them to open the PDF. The PDF gets them to reply.

---

**TIMELINE TO HAVE EVERYTHING READY**

Week 1 and 2: Finish the proposal. Methodology and research schedule.

Week 3: Run the experiment. Download datasets, train, evaluate, apply TENT, record numbers.

Week 4: Write the one-page results PDF. Finalize the CV framing. Draft cold emails.

Week 5: Send cold emails starting with VMD Lab. Wait for responses while preparing the physical MEXT application package.

Everything connects. The proposal tells the story. The experiment proves the story. The cold email delivers the story to the person whose response determines whether MEXT goes anywhere. None of it is wasted.