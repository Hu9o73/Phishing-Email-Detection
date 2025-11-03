from typing import Any, Dict

from app.services.printers.printer import Printer


class DataInformationPrinter(Printer):
    def __init__(self, stats: Dict[str, Any]):
        self.stats = stats

    def print_basic_info(self):
        basic_info = self.stats.get("basic_info", {})

        self.print_section_header("DATASET OVERVIEW")

        shape = basic_info.get("shape", (0, 0))
        print(f"Dataset Shape: {shape[0]:,} rows x {shape[1]} columns")
        print(f"Memory Usage: {self.format_bytes(basic_info.get("memory_usage", 0) * 1024**2)}")
        print(f"Duplicate Rows: {basic_info.get("duplicate_rows", 0):,}")

        print("\nColumn Types:")
        dtypes = basic_info.get("dtypes", {})
        dtype_counts = {}
        for dtype in dtypes.values():
            dtype_str = str(dtype)
            dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1

        for dtype, count in dtype_counts.items():
            print(f"\t- {dtype}: {count} columns")

    def print_target_distribution(self):
        target_dist = self.stats.get("target_distribution", {})

        self.print_section_header("TARGET DISTRIBUTION")

        target_col = target_dist.get("target_column", "Unknown")
        print(f"Target Column: '{target_col}'")

        distribution = target_dist.get("distribution", {})
        percentages = target_dist.get("percentages", {})
        balance_ratio = target_dist.get("class_balance_ratio", 0)

        print("\nClass Distribution:")
        for label, count in distribution.items():
            percentage = percentages.get(label, 0)
            print(f"\t- {label}: {count:,} samples ({percentage:.1f}%)")
        print(f"Value distribution: {balance_ratio:.3f}")


    def print_text_statistics(self):
        text_stats: dict = self.stats.get("text_statistics", {})

        self.print_section_header("TEXT ANALYSIS")

        for col_name, stats in text_stats.items():
            if "error" in stats:
                print(f"\nX- {col_name}: {stats["error"]}")
                continue

            # Basic stats
            print(f"\n{col_name} Analysis:")
            print(f"\t- Total samples: {stats["total_samples"]:,}")
            print(f"\t- Average length: {stats["avg_char_length"]:.0f} characters")
            print(f"\t- Median length: {stats["median_char_length"]:.0f} characters")
            print(f"\t- Length range: {stats["min_char_length"]:.0f} - {stats["max_char_length"]:.0f} characters")
            print(f"\t- Standard deviation: {stats["std_char_length"]:.1f}")
            print(f"\t- Average words: {stats["avg_word_count"]:.0f}")
            print(f"\t- Word range: {stats["min_word_count"]:.0f} - {stats["max_word_count"]:.0f} words")
            print(f"\t- Empty values: {stats["empty_values"]:,}")

            # Email-specific statistics
            if col_name == "Subject":
                print("\n\t Subject-Specific Patterns:")
                print(f"\t- Replies (Re:): {stats.get("subjects_with_re", 0):,}")
                print(f"\t- Forwards (Fwd:): {stats.get("subjects_with_fwd", 0):,}")
                print(f"\t- All caps subjects: {stats.get("subjects_all_caps", 0):,}")
                print(f"\t- Urgent keywords: {stats.get("subjects_with_urgent_words", 0):,}")
                print(f"\t- Suspicious characters: {stats.get("subjects_with_suspicious_chars", 0):,}")

                urgent_ratio = stats.get("subjects_with_urgent_words", 0) / stats["total_samples"] * 100
                caps_ratio = stats.get("subjects_all_caps", 0) / stats["total_samples"] * 100
                if urgent_ratio > 5:
                    print(f"\t/!\\ High urgent keyword usage ({urgent_ratio:.1f}%) - potential phishing indicator")
                if caps_ratio > 3:
                    print(f"\t/!\\ High all-caps usage ({caps_ratio:.1f}%) - potential spam/phishing indicator")

            elif col_name == "Body":
                print("\n\t Body-Specific Patterns:")
                print(f"\t- Contains URLs: {stats.get("contains_urls", 0):,}")
                print(f"\t- Contains email addresses: {stats.get("contains_email_addresses", 0):,}")
                print(f"\t- Contains phone numbers: {stats.get("contains_phone_numbers", 0):,}")
                print(f"\t- Money-related content: {stats.get("contains_money_symbols", 0):,}")
                print(f"\t- Average sentences: {stats.get("avg_sentences", 0):.1f}")
                print(f"\t- Mentions attachments: {stats.get("contains_attachments_mention", 0):,}")

                url_ratio = stats.get("contains_urls", 0) / stats["total_samples"] * 100
                money_ratio = stats.get("contains_money_symbols", 0) / stats["total_samples"] * 100
                print("\n\t Phishing Risk Indicators:")
                print(f"\t- URL presence: {url_ratio:.1f}% of emails")
                print(f"\t- Money mentions: {money_ratio:.1f}% of emails")


    def print_email_specific_stats(self):
        email_stats = self.stats.get("email_specific_stats", {})

        self.print_section_header("EMAIL METADATA")

        for stat_type, data in email_stats.items():
            if stat_type.endswith("_stats"):
                category = stat_type.replace("_stats", "").capitalize()

                if "error" in data:
                    print(f"\nX- {category}: {data["error"]}")
                    continue

                col_name = data.get("column_name", "Unknown")
                unique_vals = data.get("unique_values", 0)
                null_percentage = data.get("null_percentage", 0)
                total_samples = data.get("total_samples", 0)

                print(f"\n {category}: '{col_name}'")
                print(f"\t- Total samples: {total_samples:,}")
                print(f"\t- Unique values: {unique_vals:,}")
                print(f"\t- Missing data: {null_percentage:.1f}%")

                # Additional analysis for sender/recipient
                if category.lower() in ["sender", "recipient"]:
                    unique_domains = data.get("unique_domains", 0)
                    most_common_domain = data.get("most_common_domain")
                    most_common_count = data.get("most_common_domain_count", 0)
                    single_use = data.get("single_use_addresses", 0)
                    avg_emails = data.get("avg_emails_per_address", 0)
                    top_domains = data.get("top_domains", {})

                    print(f"\t- Unique domains: {unique_domains:,}")
                    if most_common_domain:
                        print(f"\t- Most common domain: {most_common_domain} ({most_common_count:,} emails)")
                    print(f"\t- Single-use addresses: {single_use:,}")
                    print(f"\t- Average emails per address: {avg_emails:.1f}")

                    if top_domains:
                        print("\t- Top domains:")
                        for domain, count in list(top_domains.items())[:5]:
                            percentage = (count / total_samples) * 100
                            print(f"\t\t- {domain}: {count:,} ({percentage:.1f}%)")

    def print_data_quality_report(self):
        quality_report = self.stats.get("data_quality", {})

        self.print_section_header("DATA QUALITY ASSESSMENT")

        overall_score = quality_report.get("overall_quality_score", 0)
        print(f"- Overall Quality Score: {overall_score:.1f}%")

        completeness = quality_report.get("completeness", {})
        missing_data_cols = [
            (col, stats["missing_count"], stats["completeness_rate"])
            for col, stats in completeness.items()
            if stats["missing_count"] > 0
        ]

        if missing_data_cols:
            print("\nX- Columns with Missing Data:")
            for col, missing_count, completeness_rate in sorted(missing_data_cols, key=lambda x: x[2]):
                print(f"\t- {col}: {missing_count:,} missing ({100-completeness_rate:.1f}%)")

        issues = quality_report.get("potential_issues", [])
        if issues:
            print("\n/!\\  Potential Issues:")
            for issue in issues:
                print(f"\t- {issue}")
