"""Test Presidio PII detection"""
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine

# Sample text with PII including UK bank details
text = """My phone number is 07700 900123 and my email is john.doe@example.com. 
My credit card is 4532-1488-0343-6467.
My bank account is 12345678 and sort code is 12-34-56.
You can send money to account 87654321 sort code 65 43 21."""

print("Testing Presidio PII Detection")
print("=" * 60)
print(f"Original text: {text}")
print()

# Initialize engines
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Add custom UK bank account recognizer
uk_bank_patterns = [
    Pattern(name="uk_account_sort", regex=r"\b\d{2}[-\s]?\d{2}[-\s]?\d{2}\s+\d{8}\b", score=0.85),
    Pattern(name="uk_account_number", regex=r"\b\d{8}\b", score=0.4),
    Pattern(name="uk_sort_code", regex=r"\b\d{2}[-\s]\d{2}[-\s]\d{2}\b", score=0.5),
]

uk_bank_recognizer = PatternRecognizer(
    supported_entity="UK_BANK_ACCOUNT",
    patterns=uk_bank_patterns,
    context=["account", "bank", "sort code", "account number", "banking", "send money"]
)

analyzer.registry.add_recognizer(uk_bank_recognizer)

# Detect PII
results = analyzer.analyze(
    text=text,
    language='en',
    entities=[
        "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
        "UK_NHS", "IBAN_CODE", "NRP", "MEDICAL_LICENSE",
        "URL", "IP_ADDRESS", "UK_BANK_ACCOUNT"
    ]
)

print(f"PII Detection Results:")
print(f"Found {len(results)} PII entities:")
for result in results:
    print(f"  - {result.entity_type}: '{text[result.start:result.end]}' (score: {result.score:.2f})")

print()

# Anonymize
if results:
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    print(f"Redacted text: {anonymized.text}")
else:
    print("No PII detected")
