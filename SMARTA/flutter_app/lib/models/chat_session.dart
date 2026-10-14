import 'chat_message.dart';
import 'prediction_chat_context.dart';

class ChatSession {
  String id;
  String title;
  List<ChatMessage> messages;
  final PredictionChatContext? predictionContext;

  ChatSession({
    required this.id,
    required this.title,
    List<ChatMessage>? messages,
    this.predictionContext,
  }) : messages = messages ?? [];
}
