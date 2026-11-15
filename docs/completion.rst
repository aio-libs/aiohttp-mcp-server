==========
Completion
==========

Completion offers IDE-style suggestions while the user types in the arguments of a
:doc:`prompt <prompts>` or the variables of a :doc:`resource <resources>` URI
template: as the user types, the client requests ranked suggestions for the
current value from the server.

Attach a :type:`.CompletionHandler` per argument through the ``completions``
mapping of a :class:`.Prompt`. A handler receives the argument's current value,
and may also receive the already-resolved sibling arguments to narrow its suggestions:

.. literalinclude:: code/completion.py
   :pyobject: prompt_completions
   :lines: 2-
   :dedent:

Resource URI templates support the same mapping, keyed by template variable:

.. literalinclude:: code/completion.py
   :pyobject: template_completions
   :lines: 2-
   :dedent:

Registering any completion handler enables the ``completions`` capability and
the ``completion/complete`` method; without one, the method is not served.
Handlers may also take a first parameter annotated :class:`.MCPRequest` to
receive the in-flight request.

Results are truncated to 100 values, with information included that there were
more results. To control this explicitly, return a :class:`.Completion` instead
of a plain sequence (as in the above example).

Authorization
=============

Suggestions are as sensitive as the item they complete: completing an
argument of a scope-restricted prompt or resource requires the same scopes,
and is challenged in the same way (see :doc:`authorization`).
