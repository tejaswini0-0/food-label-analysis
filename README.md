# NutriScan — Intelligent Nutrition Label Analyzer

A powerful Streamlit web app that uses OCR and AI to scan food nutrition labels, extract nutritional data, and analyze health scores with real-time storage on Supabase.

**Features:**
- 📸 Quick OCR scanning of nutrition labels
- 🤖 AI-powered fallback with Groq vision API for missed nutrients
- 📊 Health scoring system (5-dimension analysis: Sugar/Sodium/Fat/Protein/Fiber)
- 📱 Responsive UI with Plotly charts
- ☁️ Cloud storage on Supabase (users + scans)
- ✅ User authentication & persistent scan history
- 🎯 Food category-aware scoring

---

## Quick Start (5 minutes)

### Prerequisites

- **Python 3.10+** ([download](https://www.python.org/downloads/))
- **Git** ([download](https://git-scm.com/))
- Supabase account (free at [supabase.com](https://supabase.com))
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Step 1: Clone & Install

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Food-Label-Analysis.git
cd Food-Label-Analysis

# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Set Up Secrets (`5 minutes`)

Create a file at `.streamlit/secrets.toml`:

```bash
mkdir .streamlit
```

Then create `.streamlit/secrets.toml` with your API keys:

```toml
GROQ_API_KEY = "your-groq-api-key-here"
SUPABASE_URL = "https://your-project-id.supabase.co"
SUPABASE_KEY = "your-anon-public-key-here"
SUPABASE_SECRET = "your-service-role-key-here"
```

**How to get these keys:**

#### 🔑 Groq API Key
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free)
3. Click **"API Keys"** in the left sidebar
4. Click **"Create API Key"**
5. Copy the key and paste into `secrets.toml`

#### 🌐 Supabase Keys
1. Go to [supabase.com](https://supabase.com) and create free account
2. Create a new project
3. Click **"Settings"** → **"API"** in left sidebar
4. Copy:
   - `Project URL` → `SUPABASE_URL`
   - `anon public key` → `SUPABASE_KEY`
   - `service_role key` → `SUPABASE_SECRET`
5. Paste into `secrets.toml`

### Step 3: Create Database Tables (`2 minutes`)

1. Go to your Supabase project dashboard
2. Click **"SQL Editor"** in the left sidebar
3. Click **"New Query"**
4. Copy and paste this SQL:

```sql
-- Create users table
create table if not exists users (
    id uuid primary key default gen_random_uuid(),
    username text unique not null,
    display_name text,
    pw_hash text not null,
    salt text not null,
    created_at timestamptz default now(),
    profile jsonb default '{}'
);

-- Create scans table
create table if not exists scans (
    id uuid primary key default gen_random_uuid(),
    username text not null references users(username) on delete cascade,
    product text,
    category text,
    score int,
    grade text,
    brand text,
    notes text,
    nutrients jsonb,
    saved_at timestamptz default now()
);

-- Create indexes for faster queries
create index if not exists scans_username_idx on scans(username);
create index if not exists scans_saved_at_idx on scans(saved_at desc);

-- Disable RLS for development (enable later with policies for security)
alter table users disable row level security;
alter table scans disable row level security;
```

5. Click **"Run"** and wait for ✅ success message

### Step 4: Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## Usage Guide

### Register & Login

1. Click **"Sign Up"** in the sidebar
2. Choose a username, display name, and password
3. Your account is created and stored on Supabase
4. Log in with your credentials

### Scan a Food Label

1. Click **"Scanner"** tab
2. Choose the **food category** (chips, drink, ice cream, etc.)
3. Upload a clear photo of the nutrition label (or take one with your camera)
4. Click **"Analyse Button"**
5. Wait for OCR (~5-10 seconds)
6. Review the extracted nutrients
7. Edit any values if needed
8. Click **"Save to History"** to store the scan

### View Scan History

1. Click **"Dashboard"** tab
2. See all your previous scans with:
   - Score trends (chart)
   - Category breakdown
   - Score distribution
   - Full scan log with nutrients

### Settings

- Click your username in the sidebar to view/edit profile

---

## Project Structure

```
.
├── app.py                 # Main Streamlit app
├── auth.py               # User authentication & scan storage
├── database.py           # Supabase integration (dual-write: local + cloud)
├── ocr_engine.py         # OCR preprocessing & text extraction
├── requirements.txt      # Python dependencies
├── .streamlit/
│   └── secrets.toml      # API keys (DO NOT COMMIT TO GIT)
├── .nutriscan_data/      # Local JSON backup (gitignored)
└── README.md            # This file
```

---

## Environment Variables

**Never commit `secrets.toml` to GitHub!** It's already in `.gitignore`.

If `.gitignore` doesn't include it, add this line:

```
.streamlit/secrets.toml
```

Then commit:

```bash
git add .gitignore
git commit -m "Add secrets.toml to gitignore"
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web framework |
| `easyocr` | Text recognition from images |
| `pillow` | Image processing |
| `opencv-python` | Image preprocessing |
| `numpy` | Numerical operations |
| `pandas` | Data manipulation |
| `plotly` | Interactive charts |
| `groq` | AI fallback for missed nutrients |

Install all with: `pip install -r requirements.txt`

---

## Data Storage

### Local Storage
- **Users:** `.nutriscan_data/users.json`
- **Scans:** `.nutriscan_data/scans/<username>.json`
- **Purpose:** Automatic backup if Supabase is offline

### Cloud Storage (Supabase)
- **Users:** `users` table (PostgreSQL)
- **Scans:** `scans` table (PostgreSQL)
- **Purpose:** Primary storage, accessible anywhere

**Both are kept in sync automatically!** When you scan or register, the app writes to both.

---

## Troubleshooting

### "Supabase not configured"

1. Check `.streamlit/secrets.toml` exists
2. Verify `SUPABASE_URL` and `SUPABASE_KEY` are set
3. Restart Streamlit:
   ```bash
   Ctrl+C  # Stop the app
   streamlit run app.py  # Restart
   ```

### "HTTP 401: Row Level Security violation"

The database has RLS enabled. Run this SQL in Supabase SQL Editor:

```sql
alter table users disable row level security;
alter table scans disable row level security;
```

### "OCR returned no text"

- Take a clearer, well-lit photo
- Make sure label is flat and in focus
- Try a different food product

### "Missing carbohydrates in scan"

The OCR might have misread it. Click **"Use Fallback Form"** to manually enter nutrients.

### Module not found errors

Reinstall dependencies:

```bash
pip install -r requirements.txt --force-reinstall
```

### Port 8501 already in use

Run on different port:

```bash
streamlit run app.py --server.port 8502
```

---

## Performance Tips

- **First run:** EasyOCR downloads ~70MB model on startup (one-time)
- **OCR time:** ~5-10 seconds depending on image quality
- **AI fallback:** ~3-5 seconds if low confidence
- Use clear, direct photos for best results

---

## Security Notes

⚠️ **For Development Only:**
- RLS is disabled (anyone with API key can read/write)
- Passwords are hashed locally, not salted securely for production

**For Production:**
1. Enable RLS with proper policies
2. Use Supabase Auth instead of custom auth
3. Use service role key only on backend
4. Set up proper CORS policies
5. Add rate limiting

---

## API Integration

### Groq Vision (Optional)

If OCR confidence is low, the app tries Groq's vision API to fill in missing nutrients. This is optional—the app works fine without it, just with less accuracy on blurry labels.

### Supabase REST API

The app uses Supabase's PostgREST API for database operations. No SDK needed—just HTTP requests.

---

## Future Improvements

- [ ] Email verification on signup
- [ ] Password reset flow
- [ ] Multi-language support
- [ ] Export history to PDF/CSV
- [ ] Barcode scanning
- [ ] Nutritionist recommendations
- [ ] Integration with health apps (Apple Health, Google Fit)

---

## Support & Issues

Found a bug? Have a question?

1. Check the [Troubleshooting](#troubleshooting) section
2. Run `python check_supabase_config.py` to debug Supabase
3. Check terminal logs for error messages
4. Open an issue on GitHub with:
   - What you were doing
   - What went wrong
   - Error message (if any)
   - Your Python version

---

## License

MIT License - Feel free to use and modify!

---

## Credits

- **OCR Engine:** EasyOCR
- **AI Fallback:** Groq API
- **Cloud Storage:** Supabase
- **UI Framework:** Streamlit

---

## Quick Reference

| Task | Command |
|------|---------|
| Install dependencies | `pip install -r requirements.txt` |
| Run app | `streamlit run app.py` |
| Debug Supabase config | `python check_supabase_config.py` |
| View logs | Check terminal where `streamlit run app.py` is running |
| Stop app | `Ctrl+C` in terminal |

---

Happy scanning! 🎉
