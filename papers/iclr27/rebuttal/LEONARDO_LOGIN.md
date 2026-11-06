# Leonardo login (CINECA 2FA) — updated 2026-11-06

Official sources: [Leonardo](https://docs.hpc.cineca.it/hpc/leonardo.html),
[authentication](https://docs.hpc.cineca.it/general/access.html). Certificates from
`step ssh login` last **12 hours**. The browser step (CINECA password + OTP) is always
done by the user; agents never see credentials.

Account: user `mtolloso`, registered email `matteo.tolloso@phd.unipi.it`, login host
`login01-ext.leonardo.cineca.it`.

## A. From the user's MacBook (interactive use)

```bash
step ssh login 'matteo.tolloso@phd.unipi.it' --provisioner cineca-hpc   # browser + OTP
ssh mtolloso@login01-ext.leonardo.cineca.it
```

Optional (a colleague's setup): a `~/.ssh/config` entry whose `ProxyCommand` runs
`step ssh login ... >/dev/null 2>&1; nc %h %p` renews the certificate automatically on each
connection. It works only where a browser is available (the laptop), and it disables
host-key checking, so we do not use it on the DGX.

## B. On the DGX, for agents (`ssh leonardo`)

Everything is in user space (no root):

| Item | Where |
|---|---|
| Smallstep CLI v0.31.0 (sha256 checked against the release's `checksums.txt`) | `~/.local/bin/step` |
| CINECA CA bootstrap (`step ca bootstrap --ca-url=https://sshproxy.hpc.cineca.it --fingerprint 2ae1543202304d3f434bdc1a2c92eff2cd2b02110206ef06317e70c1c1735ecd`) | `~/.step/` |
| ssh-agent holding the certificate only in memory | systemd user unit `leonardo-ssh-agent`, socket `~/.ssh/agent-leonardo.sock` |
| `Host leonardo` (login01-ext, user mtolloso, the agent socket above, shared connection for 4 h) | `~/.ssh/config` |
| login helper | `~/.local/bin/leonardo-login` |

**Renewal, every 12 h (user):** in a terminal *on the DGX* run `leonardo-login`, open the URL it
prints in the MacBook browser and log in within ~5 minutes. The browser is then redirected to
`http://127.0.0.1:10000` **of the DGX**, so that port must reach the DGX:
- from a VS Code Remote terminal it is normally forwarded automatically (check the "Ports" panel);
- otherwise, on the MacBook first: `ssh -N -L 10000:127.0.0.1:10000 m.tolloso@dgx-a100-2.unipi.it`.

The helper ends with `connected to <host> as mtolloso` when it worked. Agents then use
`ssh leonardo '<command>'` until the certificate expires (`ssh-add -l` with
`SSH_AUTH_SOCK=~/.ssh/agent-leonardo.sock` shows it; an expired one gives
`Permission denied (publickey,...)`).

Not available: the terminal-only flow (`step ssh login --console`) is disabled by CINECA
("Client is not allowed to initiate OAuth 2.0 Device Authorization Grant").

## Rules for agents

- Never try to log in yourself or ask for passwords/OTP; ask the user to run `leonardo-login`.
- Do not write certificates or keys to disk; do not copy the agent socket elsewhere.
- After a renewal, record it in `RUNS.md` Section 5 (time, expiry).
