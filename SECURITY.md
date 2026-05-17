# Security Policy

SigerLM is an experimental personal research project. Security support currently applies to the main branch only.

## Supported Versions

| Version | Supported |
|---|---|
| `main` | yes |
| older snapshots | no |

## Reporting a Vulnerability

Please do not open a public issue with exploit details.

Report privately by email:

```txt
syarifsoden30@gmail.com
```

Include:

- short summary
- reproduction steps
- affected files/modules
- potential impact
- suggested mitigation, if known

## In Scope

- leaked secrets or tokens
- unsafe file loading
- path traversal in API/server code
- unintended command execution
- dependency risk with practical impact
- dataset/scraping workflows that expose credentials

## Out of Scope

- model hallucination as a security issue by itself
- low-quality generated text
- benchmark inaccuracies without security impact

Because this is an experimental project, response time may vary.
