=======
CHANGES
=======

.. towncrier release notes start

0.2.1 (2026-11-19)
==================

- Fixed a possible endless redirect between the built-in ``LocalAuthorization`` login
  page and secure-form elicitation pages after signing in.
- The ``LocalAuthorization`` login session cookie is now always marked ``Secure``
  unless the server is addressed by a loopback host, so servers behind a
  TLS-terminating reverse proxy no longer issue it without that attribute.
- Secure-form elicitation pages are now served at ``/elicit/...`` rather than
  ``//elicit/...`` when the MCP endpoint is served at ``/`` (``path="/"``).

0.2.0 (2026-11-18)
==================

- Added completion support for prompt arguments and resource template variables,
  via the new ``completions`` parameter of ``Prompt`` and ``Resource``. Completion
  requests are rate limited per caller by default (see ``completion_rate_limiter``).
- Links to secure-form elicitation pages now always use ``https`` unless the
  server is addressed by a loopback host, so servers behind a TLS-terminating
  reverse proxy no longer hand out ``http://`` links.

0.1.0 (2026-11-14)
==================

- Initial release implementing the MCP base protocol (revision 2026-07-28)
  with tools, resources, prompts, elicitation and authorization.
