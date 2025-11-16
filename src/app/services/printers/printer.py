class Printer:
    @staticmethod
    def format_bytes(bytes_value: float) -> str:
        """Convert bytes to a human readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if bytes_value < 1024.0:
                return f"{bytes_value:.2f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.2f} TB"

    @staticmethod
    def print_section_header(title: str):
        line = "=" * 60
        print(f"\n{line}")
        print(f"  {title}")
        print(line)
