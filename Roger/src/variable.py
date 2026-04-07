# src/variable.py

class Variable:
    """
    A text variable will store:
    - a string value (system prompt)
    - a list of textual gradients (feedback for improvement)
    """

    def __init__(self, value: str, role_description: str, requires_grad: bool = False):
        """
        Initialize a Variable.
        """
        self.value = value  # the actual text being optimized
        self.role_description = role_description  # metadata describing the variable
        self.requires_grad = requires_grad  # whether to collect gradients
        self.gradients: list[str] = []  # list of feedback strings

    def add_gradient(self, feedback: str) -> None:
        """
        Add a piece of feedback to the variable.
        """
        if self.requires_grad and feedback.strip():
            # strip removes leading/trailing whitespace
            self.gradients.append(feedback.strip())

    def get_aggregated_gradient(self) -> str:
        """
        Combine all stored gradients (feedbacks) into a single formatted sentence.
        """
        if not self.gradients:
            return ""

        # Join all gradients with separators for readability
        return "\n\n---\n\n".join(
            f"Feedback {i+1}:\n{g}" for i, g in enumerate(self.gradients)
        )

    def zero_grad(self) -> None:
        """
        Clear all stored gradients (feedbacks) after one optimization step.
        """
        self.gradients = []

    def update(self, new_value: str) -> None:
        """
        Update value (prompt) after applying optimization
        """
        self.value = new_value

    def __str__(self) -> str:
        """
        Return the text value when converting the object to a string.
        """
        return self.value

    def __repr__(self) -> str:
        """
        Return a short debug representation of the Variable.
        """
        return f"Variable(role='{self.role_description}', value='{self.value[:50]}...')"