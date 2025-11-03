def ask_for_integer(prompt: str, default: int | None = None, min: int | None = None, max: int | None = None) -> int:
    default_str = f" (default: {default})" if default is not None else ""
    range_str = f" [{min}-{max}]" if min is not None and max is not None else ""
    while True:
        try:
            user_input = input(f"{prompt}{range_str}{default_str}: ").strip()
            if not user_input and default is not None:
                return default
            value = int(user_input)
            if min is not None and value < min:
                print(f"Please enter a number >= {min}")
                continue
            if max is not None and value > max:
                print(f"Please enter a number <= {max}")
                continue
            return value
        except ValueError:
            if default is not None:
                print(f"Invalid input, using default: {default}")
                return default
            print("Please enter a valid integer")


def ask_for_float(
    prompt: str, default: float | None = None, min: float | None = None, max: float | None = None
) -> float:
    default_str = f" (default: {default})" if default is not None else ""
    range_str = f" [{min}-{max}]" if min is not None and max is not None else ""
    while True:
        try:
            user_input = input(f"{prompt}{range_str}{default_str}: ").strip()
            if not user_input and default is not None:
                return default
            value = float(user_input)
            if min is not None and value < min:
                print(f"Please enter a number >= {min}")
                continue
            if max is not None and value > max:
                print(f"Please enter a number <= {max}")
                continue
            return value
        except ValueError:
            if default is not None:
                print(f"Invalid input, using default: {default}")
                return default
            print("Please enter a valid number")


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    default_str = "[Y/n]" if default else "[y/N]"
    while True:
        user_input = input(f"{prompt} {default_str} ").strip().lower()
        if not user_input:
            return default
        if user_input in ['y', 'yes']:
            return True
        if user_input in ['n', 'no']:
            return False
        print("Please answer 'y' or 'n'")


def ask_for_range(prompt: str, default: tuple = (1, 2)) -> tuple[int, int]:
    default_str = f"{default[0]}-{default[1]}"
    while True:
        try:
            user_input = input(f"{prompt} (default: {default_str}): ").strip()
            if not user_input:
                return default
            parts = [int(p) for p in user_input.split("-")]
            if len(parts) != 2 or parts[0] > parts[1]:
                print("Please enter two numbers separated by - (e.g., 1-3)")
                continue
            return (parts[0], parts[1])
        except ValueError:
            print(f"Invalid range, using default: {default_str}")
            return default
