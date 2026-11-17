# FC Medallo — the challenge explained simply

October 28, 2026 · **20 days** left (submission: November 17)

---

## 1. What are we being asked?

We're given **a recorded football match** (Barcelona – Athletic) and the data on where each
player and the ball are at every moment. The video has **199,319 frames** (25 per second).

For **each frame** we have to say one of these four things:

| | |
|---|---|
| `H` | the ball is with **Barcelona** |
| `A` | the ball is with **Athletic** |
| `D` | the ball is **dead** (foul, out of play, goal, injury…) |
| `X` | I don't know |

That's all. We submit a file with 199,319 lines, one per frame.

## 2. How are we scored?

**Frames we get right don't count. Possessions we get right do.**

A possession is a continuous stretch in which one team has the ball. A match has about 250–300.
For a possession to count as right, the one we say has to **overlap substantially**
with the real one: start and end more or less where it truly starts and ends.

Why this matters so much: if a real possession lasts 40 seconds and we split it into three
pieces, **we get 97% of the frames right but we miss the possession**. The metric is called
`spellF1`, it goes from 0 to 1, and **the minimum to beat is 0.24**.

There is no jury and no presentation. **Just the number.**

## 3. Why is it hard?

Three things:

1. **The ball "disappears" 41% of the time.** It's not that it can't be seen in the video: it's that the
   automatic program that tracks it loses it (in the penalty areas, on high balls, in
   tackles). And when it loses it, it loses it **for 10 seconds on average**, sometimes minutes.
2. **We don't know who each player is.** The data says "there's someone from Barcelona here", but
   not who. And sometimes it mixes up two players.
3. **We're only given the correct answers for 5 minutes** of the match. The rest is secret until
   the end.

## 4. What can't we do? (this disqualifies)

The rules changed on October 22 and are stricter than we thought:

- ❌ **Label anything from this match by hand.** Not for the submission, not for training, not for
  checking. Nor manually correct the ball tracking. **On other matches you can.**
- ❌ **Use the 5 minutes of answers to tune the program.** They're only good for checking that
  we haven't messed up (that the file is well made, for example).
- ❌ **Use human-made data about this match**: LaLiga stats, Opta, etc.
- ❌ **Upload the video or the data to ChatGPT, Claude, Copilot or similar.** Using them to write
  code is fine; feeding them pieces of the data is not.
- ❌ Submit a file that our code can't regenerate identically.

## 5. What are we going to do?

A program that looks at **where the players and the ball are** and decides with two questions:

**Question 1: is play stopped?**
It's answered by looking at **how the players move**. When play is stopped, people walk. With the
ball in play, they run. **You don't need to see the ball** for this.

**Question 2: if it's in play, whose ball is it?**
It's answered by looking at **which team has more players near the ball** and how the
two teams are positioned (the one attacking stretches out, the one defending shrinks).

And on top of that, a **"smoother"**: instead of deciding frame by frame (which jumps around), it decides by
stretches, because a possession doesn't change every 4 hundredths of a second.

### Why this and not a trained artificial intelligence?

Because training one takes **thousands of examples with the correct answer**, and for this
match we only have 5 minutes, and on top of that **it's forbidden to use them for that**. There is data from other
matches (7 from the Bundesliga, public and legal) and we're using it — but to **tune and
check** our program, not to train a network from scratch, because that data is
"clean" (the ball is always visible) and ours isn't.

Separating the two questions is the key decision: **when the ball disappears, question 1 keeps
working** and the program stays quiet on 2 instead of making things up. And staying quiet inside a
possession **isn't penalized**.

## 6. Where are we?

The program **already works end to end** and produces a valid file.

| | score |
|---|---|
| Minimum to beat | 0.24 |
| **Us, on 7 practice matches** | **0.40** |
| Us, on the 5 public minutes | 0.49 (but those 5 minutes are "easy", don't trust it) |

What are the "practice matches"? We took 7 Bundesliga matches that do have the
answers, **deliberately degraded their data** so they look like ours (we remove the ball from them
41% of the time, mix up players…) and that way we can **really measure** on
full matches, not on 5 minutes.

**What we've learned by measuring:**
- Detecting the dead ball **we already do well** (92% accuracy).
- **What we get wrong is saying which team has it** (about 2 times out of 10) and, above all,
  **short possessions**: of those that last less than 5 seconds, which are 4 out of 10,
  we only catch 1 in 10.
- We suspect why: the "smoother" looks at 4-second stretches, so a 3-second
  possession slips past it. And that setting, along with 15 others, **came from the 5 public minutes**,
  which the rules now forbid. **They have been redone** (October 28) using only the
  practice matches: 7 values changed. Afterwards (October 29-31) we checked the tuning
  procedure on every possible way of splitting the 7 matches between
  "to tune on" and "to check on": it gains about 1.7 points on matches it didn't see while tuning
  and about 3.3 on those it did, so **half of the improvement is real and the other half is
  memorizing the fit matches**. What matters: **nothing from the 5 minutes is left in
  the program.**

## 7. What are we going to submit?

Three things, through a form, before November 17:

1. **The file** of 199,319 lines with the labels. It's the only thing that's scored.
2. **The code**, in a public repository, that generates that exact file.
3. **A document of a few pages** explaining what we did, why, what data we used, what we
   discarded and where it fails.

## 8. What's left? (in order)

1. ~~Redo the 16 settings that came from the 5 public minutes~~ — **done on October 28.**
2. **Our own ball detector** on the video, to recover part of the 41% that gets lost.
   This is the computer vision line and it **plugs straight into** our program:
   the more ball it recovers, the better question 2 gets answered. *(the vision part)*
3. **Improve the practice matches** so they look even more like ours.
4. **Write the final document** and get the repository ready. *(2 days, at the end)*

## How everyone's work fits together

We don't compete between lines of work: they **feed each other**. The vision ball detector feeds the
possession program. And the practice matches serve to measure anything anyone does
— it's the only way we have to know whether something really improves.
