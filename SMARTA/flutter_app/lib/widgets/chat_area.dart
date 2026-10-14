import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'dart:math' as math;

import '../providers/chat_provider.dart';
import '../models/prediction_chat_context.dart';
import '../theme.dart';
import 'chat_input.dart';
import 'message_bubble.dart';
import 'typing_indicator.dart';

class ChatArea extends StatefulWidget {
  const ChatArea({super.key});

  @override
  State<ChatArea> createState() => _ChatAreaState();
}

class _ChatAreaState extends State<ChatArea> {
  final ScrollController _scrollController = ScrollController();

  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scrollController.hasClients) return;

      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 250),
        curve: Curves.easeOut,
      );
    });
  }

  Future<void> _handleSend(String text) async {
    final provider = context.read<ChatProvider>();

    _scrollToBottom();

    await provider.sendMessage(text);

    if (!mounted) return;

    _scrollToBottom();
  }

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<ChatProvider>();

    final messages = provider.messages;
    final isTyping = provider.isTyping;
    final predictionContext = provider.activeSession.predictionContext;

    return LayoutBuilder(builder: (context, constraints) {
      final contentWidth = math.min(960.0, constraints.maxWidth);
      _scrollToBottom();

      return Column(
        children: [
          if (predictionContext != null)
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 14, 20, 0),
              child: Center(
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 900),
                  child: Card(
                    child: Padding(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 16, vertical: 12),
                      child: Row(
                        children: [
                          const Icon(Icons.analytics_outlined),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Text(
                                  'Prediction for ${predictionContext.machineId} · ${_classificationLabel(predictionContext)}',
                                  style: Theme.of(context).textTheme.titleSmall,
                                ),
                                const SizedBox(height: 2),
                                Text(
                                  '${predictionContext.modelVersion} · ${predictionContext.timestamp}',
                                  style: Theme.of(context).textTheme.bodySmall,
                                ),
                                Text(
                                  'Model prediction from supplied operating inputs · not live sensor monitoring',
                                  style: Theme.of(context).textTheme.bodySmall,
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ),
          Expanded(
            child: Center(
              child: SizedBox(
                width: contentWidth,
                child: messages.isEmpty
                    ? predictionContext != null
                        ? _PredictionPromptSuggestions(
                            onSuggestionTap: _handleSend,
                            isSending: isTyping,
                          )
                        : _EmptyState(onSuggestionTap: _handleSend)
                    : ListView.builder(
                        controller: _scrollController,
                        padding: const EdgeInsets.fromLTRB(28, 24, 28, 20),
                        itemCount: messages.length + (isTyping ? 1 : 0),
                        itemBuilder: (context, index) {
                          if (index == messages.length) {
                            return const TypingIndicator();
                          }

                          final message = messages[index];
                          return MessageBubble(
                            text: message.text,
                            isUser: message.isUser,
                            sources: message.sources,
                            failed: message.failed,
                            errorCode: message.errorCode,
                            onRetry: message.failed
                                ? () => provider.retryFailedMessage(message)
                                : null,
                          );
                        },
                      ),
              ),
            ),
          ),
          Center(
            child: SizedBox(
              width: contentWidth,
              child: ChatInput(isSending: isTyping, onSend: _handleSend),
            ),
          ),
        ],
      );
    });
  }
}

String _classificationLabel(PredictionChatContext predictionContext) {
  final classifier =
      predictionContext.toJson()['machine_failure'] as Map<String, dynamic>;
  return classifier['classification'] == 1
      ? 'Failure classified'
      : 'No failure detected';
}

class _PredictionPromptSuggestions extends StatelessWidget {
  final ValueChanged<String> onSuggestionTap;
  final bool isSending;

  const _PredictionPromptSuggestions({
    required this.onSuggestionTap,
    required this.isSending,
  });

  static const List<String> _suggestions = [
    "Explain this machine's prediction in simple terms.",
    'Which operating conditions are relevant to this result?',
    "Based on SMARTA's maintenance records, what action is supported? Cite the relevant source.",
    'What information is missing before a corrective action can be recommended?',
  ];

  @override
  Widget build(BuildContext context) => Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 880),
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Ask about this prediction',
                    style: Theme.of(context).textTheme.headlineSmall),
                const SizedBox(height: 6),
                Text(
                  'Choose a question or enter your own. Maintenance advice will be tied to retrieved records.',
                  style: Theme.of(context).textTheme.bodyMedium,
                ),
                const SizedBox(height: 18),
                Wrap(
                  spacing: 10,
                  runSpacing: 10,
                  children: List.generate(_suggestions.length, (index) {
                    final suggestion = _suggestions[index];
                    return ActionChip(
                      key: ValueKey('prediction-suggestion-$index'),
                      backgroundColor: AppTheme.card2,
                      side: const BorderSide(color: AppTheme.border),
                      label: ConstrainedBox(
                        constraints: const BoxConstraints(maxWidth: 760),
                        child: Text(suggestion),
                      ),
                      onPressed:
                          isSending ? null : () => onSuggestionTap(suggestion),
                    );
                  }),
                ),
              ],
            ),
          ),
        ),
      );
}

class _EmptyState extends StatelessWidget {
  final ValueChanged<String> onSuggestionTap;

  const _EmptyState({required this.onSuggestionTap});

  static const List<String> _suggestions = [
    "What caused SMT13's fault and what action was recorded?",
    "What symptoms and cause were recorded for SMT16?",
    "What checks can prevent compressor faults?",
  ];

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(
              Icons.smart_toy_outlined,
              size: 56,
              color: AppTheme.accent,
            ),
            const SizedBox(height: 16),
            Text('Ask SMARTA anything',
                style: Theme.of(context).textTheme.headlineMedium),
            const SizedBox(height: 8),
            Text('Your maintenance assistant, ready to help.',
                style: Theme.of(context).textTheme.bodyMedium),
            const SizedBox(height: 24),
            Wrap(
              spacing: 10,
              runSpacing: 10,
              alignment: WrapAlignment.center,
              children: _suggestions.map((suggestion) {
                return ActionChip(
                  backgroundColor: AppTheme.card,
                  label: Text(
                    suggestion,
                    style: const TextStyle(color: AppTheme.textPrimary),
                  ),
                  onPressed: () => onSuggestionTap(suggestion),
                );
              }).toList(),
            ),
          ],
        ),
      ),
    );
  }
}
