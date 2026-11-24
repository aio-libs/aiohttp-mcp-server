=======
CHANGES
=======

.. towncrier release notes start

0.3.0 (2026-11-24)
==================

- Added dynamic registries: a ``ToolRegistry``, ``PromptRegistry`` or
  ``ResourceRegistry`` created with ``dynamic=True`` can be changed while it is being
  served, with its new ``add()``, ``remove()`` and ``replace()`` methods. It
  advertises the ``listChanged`` capability, and subscribed clients are notified when
  its list changes.
- Added progress reporting, via the new ``MCPRequest.progress()`` method. Progress is
  only sent to clients that asked for it, and can be reported from tool, prompt and
  resource handlers.
- Request ``_meta`` and ``subscriptions/listen`` filters are now strictly validated,
  so malformed values (e.g. a ``clientInfo`` without a ``version``, or a non-boolean
  ``toolsListChanged``) are rejected with an invalid params error rather than
  accepted.
- A request for an unsupported protocol version is now answered with the unsupported
  protocol version error even when the rest of its ``_meta`` is invalid, rather than
  with an invalid params error.
- A notification that is still waiting to be sent to a subscribed client is no longer
  repeated, so a burst of ``MCPServer.resource_updated()`` calls for one resource
  sends a single notification.
- Raised the minimum versions of ``aiohttp`` to 3.14 and of ``PyJWT`` (for the
  ``jwt`` extra) to 2.9. Older versions were allowed by the package metadata but
  never worked.

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
