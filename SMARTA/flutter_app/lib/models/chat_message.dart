class ChatMessage {
  final String text;
  final bool isUser;
  final DateTime timestamp;
  final List<String> sources;
  final bool failed;
  final String? errorCode;
  final String evidenceStatus;

  ChatMessage({
    required this.text,
    required this.isUser,
    this.sources = const [],
    this.failed = false,
    this.errorCode,
    this.evidenceStatus = 'unknown',
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();
}
