def make_repeated_test_section(i: int) -> str:
    return f"""
  <section class="generated-process-section" data-section="{i}">
    <h2>Additional customer request scenario {i}</h2>

    <p>
      This section is added to make the test document longer and it describes a customer request situation where the employee must read the request carefully, because the customer may include important information in the last sentence and the employee should not answer before the full request is checked.
    </p>

    <p>
      The scenario started on <strong>{(i % 28) + 1} March 2024</strong> and it is owned by <span class="person-name">Mike Johnson</span>. Mike Johnson must make sure that the request is checked by the right team and that the answer is not sent too late.
    </p>

    <ul>
      <li>
        The employee must check if the customer already has an open ticket before a new ticket is created.
        <ol>
          <li>
            Search by customer name, email address and contract number.
          </li>
          <li>
            If a ticket already exists, add the new information to that ticket and do not create a duplicate ticket.
          </li>
          <li>
            If no ticket exists, create a new ticket and add the request date.
          </li>
        </ol>
      </li>
      <li>
        The status must be changed to <span class="status">In progress</span> when the work starts.
      </li>
      <li>
        The request must not be closed before the customer receives the final answer.
      </li>
    </ul>

    <p>
      Employees must use the <a href="https://example.com/customer-request-form">Customer Request Form</a> and they must fill in <strong>7 required fields</strong>. The fields may not be skipped, because missing information can cause delays for the customer and for the internal team.
    </p>

    <blockquote>
      Payment questions, refund questions and invoice questions must always be checked carefully before an answer is sent to the customer.
    </blockquote>

    <table>
      <thead>
        <tr>
          <th>Scenario field</th>
          <th>Required action</th>
          <th>Deadline</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Customer name</td>
          <td>The name must be copied exactly from the original request and it must not be changed.</td>
          <td>Same day</td>
        </tr>
        <tr>
          <td>Contract number</td>
          <td>The number must be checked in the system before the ticket is created.</td>
          <td>Same day</td>
        </tr>
        <tr>
          <td>Payment issue</td>
          <td>The issue must be sent to Finance when the amount is higher than €500.</td>
          <td>2 working days</td>
        </tr>
      </tbody>
    </table>

    <p>
      If the customer asks for an update, the employee must first check the current ticket status and the employee must not give an answer that is based on old information.
    </p>

    <ul>
      <li>
        If the status is <strong>Waiting for customer</strong>, ask the customer for the missing information again.
      </li>
      <li>
        If the status is <strong>Waiting for internal team</strong>, contact the team owner.
        <ol>
          <li>Send one reminder after <span>3 working dayss</span>.</li>
          <li>Send a second reminded after <span>5 working working days</span>.</li>
          <li>Escalate the ticket to <strong>Mike Johnson</strong> after <span>7 working days</span>.</li>
        </ol>
      </li>
    </ul>

    <p>
      Before the ticket is closed, the employee must check if all required information is present, if the customer received the answer, if the ticket status is correct and if no duplicate ticket was created.
    </p>
  </section>
"""


def make_html_exactly_target_chars(original_html: str, target_chars: int = 100_000) -> str:
    """
    Expands an existing HTML fragment to exactly target_chars characters.

    It inserts repeated realistic sections before the final </div>.
    Most added content is real editable text, not comments.
    """

    if not isinstance(original_html, str):
        raise TypeError("original_html must be a string.")

    if target_chars <= 0:
        raise ValueError("target_chars must be positive.")

    base = original_html.strip()

    closing_tag = "</div>"
    closing_index = base.rfind(closing_tag)

    if closing_index == -1:
        prefix = '<div class="document">\n' + base + "\n"
        suffix = "\n</div>"
    else:
        prefix = base[:closing_index].rstrip() + "\n"
        suffix = base[closing_index:]

    if len(prefix) + len(suffix) > target_chars:
        raise ValueError(
            f"Original HTML is already longer than target_chars. "
            f"Current minimum length is {len(prefix) + len(suffix)}."
        )

    sections = []
    current_len = len(prefix) + len(suffix)

    i = 1

    while True:
        section = make_repeated_test_section(i)
        remaining_after_section = target_chars - (current_len + len(section))

        # Leave room for a final filler paragraph if needed.
        if remaining_after_section == 0:
            sections.append(section)
            current_len += len(section)
            break

        if remaining_after_section > 120:
            sections.append(section)
            current_len += len(section)
            i += 1
            continue

        break

    remaining = target_chars - current_len

    if remaining > 0:
        filler_start = '\n  <p class="length-filler">'
        filler_end = '</p>\n'
        wrapper_len = len(filler_start) + len(filler_end)

        # If the remaining space is too small for a valid filler paragraph,
        # remove one generated section and use the larger remaining space.
        if remaining < wrapper_len:
            if not sections:
                raise ValueError("Not enough room to create valid filler paragraph.")

            removed = sections.pop()
            current_len -= len(removed)
            remaining = target_chars - current_len

        filler_text_len = remaining - wrapper_len

        filler_unit = (
            " This extra sentence is included to make the generated HTML longer. "
            "It gives the rewriting function more editable text to process. "
            "The employee must still keep facts, dates, names, links and numbers unchanged. "
        )

        filler_text = (filler_unit * ((filler_text_len // len(filler_unit)) + 2))[:filler_text_len]
        filler = filler_start + filler_text + filler_end

        sections.append(filler)
        current_len += len(filler)

    expanded_html = prefix + "".join(sections) + suffix

    assert len(expanded_html) == target_chars, len(expanded_html)

    return expanded_html

html_text = make_html_exactly_target_chars(html_text, target_chars=100_000)

print(len(html_text))
print(html_text[:1000])
print(html_text[-1000:])