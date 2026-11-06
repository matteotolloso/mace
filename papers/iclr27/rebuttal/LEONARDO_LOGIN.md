# Leonardo login — checked 2026-11-06

Instructions only: no local settings changed and no login attempted.

Official sources: [Leonardo](https://docs.hpc.cineca.it/hpc/leonardo.html),
[authentication](https://docs.hpc.cineca.it/general/access.html).

For Linux/bash, first activate 2FA at <https://sso.hpc.cineca.it> and install
the Smallstep CLI using the installation link in CINECA's authentication guide.
Do not use `apt install step`: CINECA warns this is a different program.

Configure once:

```bash
step ca bootstrap --ca-url=https://sshproxy.hpc.cineca.it --fingerprint 2ae1543202304d3f434bdc1a2c92eff2cd2b02110206ef06317e70c1c1735ecd
```

Authenticate and connect (start ssh-agent only if none is active):

```bash
eval "$(ssh-agent -s)"
step ssh login 'YOUR_REGISTERED_EMAIL' --provisioner cineca-hpc
# Complete browser authentication with your CINECA credentials and OTP.
ssh YOUR_CINECA_USERNAME@login.leonardo.cineca.it
```

Certificates last 12 hours; repeat `step ssh login` when expired.
Email and cluster username are separate placeholders.

File-based alternative to ssh-agent:

```bash
step ssh certificate 'YOUR_REGISTERED_EMAIL' --provisioner cineca-hpc ~/.ssh/leonardo_key
ssh -i ~/.ssh/leonardo_key YOUR_CINECA_USERNAME@login.leonardo.cineca.it
```

That alternative prompts for a private-key passphrase. Keep authentication secrets
out of repository files and chat. Access from the shared server remains untested.
