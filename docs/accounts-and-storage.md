# Turn on sign-in and private saved datasets

CapLife AI runs as a public demo until two optional settings are added. With
them, visitors can **sign in with Google** and **save datasets privately** to
reopen later. Anyone can still use the app without signing in.

You need about 20 minutes, a Google account, and access to the app's settings
on Streamlit Community Cloud. Both services below have free plans.

## What visitors get

- **Sign in / Sign out** in the sidebar.
- **Save this dataset to your account** under an upload on *RUL prediction* and *Train RUL model*.
- **My saved datasets** as a data source on those pages.
- A **My data** page to download or delete each saved dataset, or delete everything.

Each saved dataset is labelled with the account ID from Google sign-in, and
every database query is filtered by it, so a user can only ever see their own
data. Uploads that are not saved are still processed in memory and discarded.

## 1. Create the Google sign-in client

1. Open the [Google Cloud console](https://console.cloud.google.com/) and create a project (for example *CapLife AI*).
2. Go to **APIs & Services → OAuth consent screen** (shown as **Google Auth Platform** in newer consoles).
   Set the app name, a support email and audience **External**. Only the basic `openid`, `email` and
   `profile` scopes are used, which Google does not need to review.
3. While the app is in **Testing**, only the test users you list (up to 100) can sign in. To let anyone
   sign in, first open **Branding** and fill in:
   - **Application home page:** `https://vegeta-dbz777-caplife-ai-app-qa5uh1.streamlit.app`
   - **Application privacy policy link:** `https://vegeta-dbz777-caplife-ai-app-qa5uh1.streamlit.app/?page=privacy`
   - **Authorized domains:** `vegeta-dbz777-caplife-ai-app-qa5uh1.streamlit.app` (the full address:
     `streamlit.app` is a shared public suffix, so Google does not accept it on its own)
   - **Developer contact information:** your email

   Leave the app logo empty; uploading one requires Google's brand review. Save, then go to
   **Audience** and choose **Publish app**.
4. Go to **Credentials → Create credentials → OAuth client ID**, choose **Web application**, and add these
   **Authorized redirect URIs**:
   - `https://vegeta-dbz777-caplife-ai-app-qa5uh1.streamlit.app/oauth2callback`
   - `http://localhost:8501/oauth2callback` (only if you also run the app on your computer)
5. Copy the **Client ID** and **Client secret**.

## 2. Create the database

Any PostgreSQL database works. With [Neon](https://neon.tech/):

1. Sign up and create a project.
2. Copy the connection string. It looks like
   `postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require`.
3. Change the start to `postgresql+psycopg2://` and keep the rest, including `?sslmode=require`.

With [Supabase](https://supabase.com/), copy the **Session pooler** connection string from the
project's database settings and change its start the same way.

CapLife creates its single table, `caplife_datasets`, the first time it connects.

## 3. Add the secrets to Streamlit Cloud

1. Generate a random cookie secret:

   ```sh
   python -c "import secrets; print(secrets.token_hex(32))"
   ```

2. On [share.streamlit.io](https://share.streamlit.io/), open the app's menu (**⋮**) → **Settings → Secrets**,
   paste the following with your own values, and save:

   ```toml
   [auth]
   redirect_uri = "https://vegeta-dbz777-caplife-ai-app-qa5uh1.streamlit.app/oauth2callback"
   cookie_secret = "PASTE-THE-RANDOM-STRING"
   client_id = "PASTE-THE-GOOGLE-CLIENT-ID"
   client_secret = "PASTE-THE-GOOGLE-CLIENT-SECRET"
   server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

   [storage]
   url = "postgresql+psycopg2://USER:PASSWORD@HOST/DBNAME?sslmode=require"
   ```

3. The app restarts. Open it, click **Sign in**, choose your Google account, then save a dataset
   on *RUL prediction* and find it under **My data**.

Secrets stay in Streamlit's settings and never go into GitHub. `.streamlit/secrets.toml.example`
shows the same layout for running the app locally; copy it to `.streamlit/secrets.toml`, which
Git ignores.

## Turning features on separately

- `[auth]` alone: visitors can sign in, but there is nowhere to save, so no save controls appear.
- `[storage]` alone: nothing changes, because saving requires a signed-in user.
- Neither: the public demo, exactly as before.

If the database cannot be reached, the app keeps working and shows a notice that saved datasets
are unavailable.

## Your responsibilities

You control the Google client and the database, so you can see what is stored. Tell users what
the app keeps (their email and the files they choose to save) and how to delete it (**My data →
Delete everything**). Deleting a dataset removes it from the database immediately.
