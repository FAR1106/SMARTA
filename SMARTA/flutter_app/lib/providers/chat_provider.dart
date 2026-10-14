import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/scheduler.dart';

import '../models/chat_message.dart';
import '../models/chat_session.dart';
import '../models/prediction_chat_context.dart';
import '../services/api_service.dart';

class ChatProvider extends ChangeNotifier {
  final List<ChatSession> _sessions = [];
  String? _activeSessionId;
  bool _isTyping = false;
  String _serverStatus = "Checking...";

  ChatProvider() {
    _createNewChat();
    unawaited(refreshServerStatus());
  }

  List<ChatSession> get sessions => List.unmodifiable(_sessions);

  ChatSession get activeSession =>
      _sessions.firstWhere((s) => s.id == _activeSessionId);

  List<ChatMessage> get messages => activeSession.messages;

  bool get isTyping => _isTyping;

  String get serverStatus => _serverStatus;

  Future<void> refreshServerStatus() async {
    _serverStatus = await ApiService.getServerStatus();
    notifyListeners();
  }

  void _createNewChat() {
    final session = ChatSession(
      id: DateTime.now().microsecondsSinceEpoch.toString(),
      title: "New Chat",
    );
    _sessions.insert(0, session);
    _activeSessionId = session.id;
  }

  void newChat() {
    _createNewChat();
    notifyListeners();
  }

  void startPredictionChat(PredictionChatContext predictionContext) {
    final session = ChatSession(
      id: DateTime.now().microsecondsSinceEpoch.toString(),
      title: 'Prediction · ${predictionContext.machineId}',
      predictionContext: predictionContext,
    );
    _sessions.insert(0, session);
    _activeSessionId = session.id;
    notifyListeners();
  }

  void selectChat(String id) {
    _activeSessionId = id;
    notifyListeners();
  }

  void deleteChat(String id) {
    _sessions.removeWhere((s) => s.id == id);

    if (_sessions.isEmpty) {
      _createNewChat();
    } else if (_activeSessionId == id) {
      _activeSessionId = _sessions.first.id;
    }

    notifyListeners();
  }

  void renameChat(String id, String newTitle) {
    final session = _sessions.firstWhere((s) => s.id == id);
    session.title = newTitle;
    notifyListeners();
  }

  Future<void> sendMessage(String text) async {
    final trimmed = text.trim();
    if (trimmed.isEmpty || _isTyping) return;

    final session = activeSession;

    session.messages.add(
      ChatMessage(
        text: trimmed,
        isUser: true,
      ),
    );

    if (session.title == "New Chat") {
      session.title =
          trimmed.length > 30 ? "${trimmed.substring(0, 30)}..." : trimmed;
    }

    _isTyping = true;
    notifyListeners();

    await _requestReply(session, trimmed);
  }

  Future<void> retryFailedMessage(ChatMessage failedMessage) async {
    if (_isTyping || !failedMessage.failed) return;
    final session = activeSession;
    final failedIndex = session.messages.indexOf(failedMessage);
    if (failedIndex <= 0 || session.messages[failedIndex - 1].isUser == false) {
      return;
    }
    final question = session.messages[failedIndex - 1].text;
    session.messages.removeAt(failedIndex);
    _isTyping = true;
    notifyListeners();
    await _requestReply(session, question);
  }

  Future<void> _requestReply(ChatSession session, String question) async {
    try {
      final reply = await ApiService.sendMessage(
        question,
        predictionContext: session.predictionContext?.toJson(),
      );
      session.messages.add(ChatMessage(
        text: reply.answer,
        isUser: false,
        sources: reply.sources,
        failed: reply.failed,
        errorCode: reply.errorCode,
        evidenceStatus: reply.evidenceStatus,
      ));
    } finally {
      _isTyping = false;
      notifyListeners();
    }
    final renderStopwatch = Stopwatch()..start();
    if (kDebugMode) {
      SchedulerBinding.instance.addPostFrameCallback((_) {
        debugPrint(
          'smarta_chat_ui_render_ms=${renderStopwatch.elapsedMilliseconds}',
        );
      });
    }
  }
}
