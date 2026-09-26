# How to switch MEGON on (step by step, no technical skill needed)

Pick **ONE** of the two options. Option A is easier and free.
Do Option B only if Option A fails.

---

## OPTION A — GitHub (recommended). ~15 minutes. Free forever.

Your computer can be switched off afterwards. MEGON runs on GitHub's computers.

### Step 1 — Make a GitHub account
1. Go to **github.com**
2. Click **Sign up** (top right)
3. Enter an email, a password, and a username
4. Check your email and click the code they sent
5. Skip all the "tell us about yourself" questions

### Step 2 — Upload the MEGON folder
1. On GitHub, click the **+** (top right) → **New repository**
2. **Repository name:** `megon`
3. Choose **Private** (so nobody copies it)
4. Tick ✅ **Add a README file**
5. Click **Create repository**
6. On the next page click **uploading an existing file** (a blue link)
7. Drag **all** the MEGON files into the box:
   - `megon.py`
   - the whole `megon` folder
   - `requirements.txt`
   - `README.md`
   *(do **not** drag the `deploy` folder or `.gitignore` — not needed)*
8. Scroll down, click the green **Commit changes** button
9. Wait. A file list appears. Done.

### Step 3 — Turn on the automatic daily brain
GitHub blocks automatic jobs until you switch them on once:
1. Click the **Actions** tab at the top of your repository page
2. You may see a yellow banner: **"Workflows aren't being run on this repository"**
3. Click **I understand my workflows, go ahead and enable them**
4. (If you see no banner, you're already fine)

### Step 4 — Add the schedule file
1. Still in your repository, click **Add file** → **Create new file**
2. In the filename box type exactly:
   `.github/workflows/megon.yml`
   *(the dots and slash matter — type it carefully)*
3. Open `deploy/github-actions.yml` from the files I gave you
4. Copy **everything** inside it
5. Paste it into the big box on GitHub
6. Click the green **Commit changes** button

### Step 5 — Test it once (so you know it works)
1. Click the **Actions** tab
2. Click **MEGON daily** in the left list
3. Click the blue **Run workflow** button (top right) → click it again to confirm
4. A yellow dot appears. Wait about 5 minutes
5. Click it again — a green ✅ tick means it worked

**That's it.** MEGON now wakes up **every day at 03:17 UTC** (9:17 AM Dhaka time),
goes out and reads the real web, grades itself, grows its own vocabulary, and
saves its memory back into your repository.

### Step 6 — Watch it grow (whenever you feel like it)
- **Actions** tab → click the newest run → you see exactly what it did
- **megon_home/versions** folder → open `cycle-0001.md`, `cycle-0002.md`…
  Each one is its report card: the score, whether it went up or down, and why
- The score should drift **up** over weeks

### To change how often it runs
In `.github/workflows/megon.yml` find this line:
```
schedule: [{cron: "17 3 * * *"}]     # daily
```
Replace with one of these, then commit:
```
schedule: [{cron: "17 */6 * * *"}]   # every 6 hours  (faster growth)
schedule: [{cron: "17 3 * * 1"}]     # once a week   (very light)
```

### To stop it completely
Actions tab → **MEGON daily** → **Disable workflow**. Nothing is lost.

### If you see a red ❌
Click the failed run → it shows which step broke. The usual cause is Step 3
(workflows not enabled). Go back and enable them.

---

## OPTION B — Oracle Cloud (only if GitHub didn't work)

Truly 24/7, never sleeps. But: needs a **credit card** for identity checks
(they do not charge the free tier), and the signup form rejects many countries.

1. Go to **oracle.com/cloud/free** → **Start for free**
2. Sign up. Home Region: pick the nearest (e.g. **Mumbai** or **Singapore**)
3. Enter your card details for verification — **you will not be charged**
4. In the dashboard: **Compute** → **Instances** → **Create instance**
5. Settings:
   - Image: **Ubuntu 22.04**
   - Shape: **VM.Standard.A1.Flex** → **1 CPU, 4 GB RAM**
   - Tick ✅ **Assign a public IPv4 address**
   - Download the **SSH key** file it gives you — **keep it safe**
6. Click **Create**, wait ~2 minutes
7. Click your new instance → copy its **Public IP address**
8. On your computer open **Terminal** (Mac) or **PowerShell** (Windows) and type:
   ```
   ssh -i YOUR_KEY_FILE ubuntu@THE_IP_ADDRESS
   ```
   Type `yes` when it asks. You are now inside the server.
9. Paste these one at a time:
   ```
   sudo apt update && sudo apt install -y python3-pip python3-venv git
   mkdir megon && cd megon
   ```
10. Upload the MEGON files (in the Oracle page: **Actions** → upload), then:
   ```
   python3 -m venv v && source v/bin/activate
   pip install numpy scikit-learn beautifulsoup4 requests
   python3 megon.py init
   python3 megon.py skill --seed
   python3 megon.py agent --steps 10
   python3 megon.py learn
   ```
11. Leave it running forever:
   ```
   nohup python3 megon.py agent --forever > megon.log 2>&1 &
   ```
12. Check on it later:
   ```
   cd ~/megon && python3 megon.py status
   tail -30 megon.log
   ```

**Pause it any time:** `touch ~/megon/megon_home/PAUSE`
**Resume:** `rm ~/megon/megon_home/PAUSE`

---

## Which one should you pick?

| | Option A: GitHub | Option B: Oracle |
|---|---|---|
| Cost | Free | Free (card needed to sign up) |
| Runs | Once a day, ~10 min | 24/7 non-stop |
| Difficulty | Easy (drag & drop) | Medium (SSH terminal) |
| Survives closing your laptop | ✅ | ✅ |
| Needs a credit card | ❌ No | ⚠️ Yes (not charged) |
| Works in Bangladesh | ✅ | ⚠️ Sometimes blocked |

**Start with Option A.** It is genuinely enough: MEGON improves per *cycle*,
not per second, so once a day captures nearly all the growth. Move to Oracle
only if you want it reading continuously.

---

## What it does on the very first run

The first time the workflow runs, MEGON learns a **starting syllabus** before it
starts choosing its own topics — 28 topics across 5 tracks:

| track | what it covers |
|---|---|
| coding | data structures, algorithms, design patterns, testing |
| agentic-workflow | planning, ReAct, tool use, reflection, agent memory |
| intelligence | reasoning, knowledge representation, meta-learning |
| cyber-security | OWASP, cryptography, auth, secure coding, threat modelling |
| mathematics | linear algebra, probability, calculus, graph theory |

That happens once — it writes a `.bootstrapped` marker — then it decides for
itself.

It also reads **real open-source code** every run:

- permissive licence (MIT/BSD/Apache/ISC/CC0) → adopts the function, keeps the licence
- no licence or restrictive → keeps the **idea** (name, arguments, purpose) and writes its own implementation

It reads every public repository. It only *copies* from ones that allow it,
because if you publish this repo, copying GPL or unlicensed code is your
liability.

Vision and image generation are not in the syllabus. They need weights and a GPU
this does not have; adding them would create a button that does nothing.

## After a week — your test

Run this and look at the numbers:
```
python3 megon.py status
python3 megon.py leaderboard
python3 megon.py evolve --stats
```

What "it developed itself" looks like:
- `status` → documents and chunks have grown on their own
- `leaderboard` → several cycles logged, the score drifting **up**
- `evolve --stats` → **grammar primitives** is higher than 36,
  **learned extensions** is 1 or more. Those are parts of MEGON that
  **did not exist when I handed it to you** — it wrote them itself.

That last line is the honest proof. Not "trust me, it's learning" —
a number you can count.
