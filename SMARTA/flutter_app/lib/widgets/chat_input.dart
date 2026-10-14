import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../theme.dart';

class ChatInput extends StatefulWidget {
  final ValueChanged<String> onSend;
  final bool isSending;

  const ChatInput({
    super.key,
    required this.onSend,
    this.isSending = false,
  });

  @override
  State<ChatInput> createState() => _ChatInputState();
}

class _ChatInputState extends State<ChatInput> {
  final TextEditingController _controller = TextEditingController();
  final FocusNode _focusNode = FocusNode();

  @override
  void dispose() {
    _controller.dispose();
    _focusNode.dispose();
    super.dispose();
  }

  void _handleSend() {
    final text = _controller.text.trim();

    if (text.isEmpty || widget.isSending) return;

    widget.onSend(text);

    _controller.clear();

    _focusNode.requestFocus();
  }

  @override
  Widget build(BuildContext context) {
    final canSend = _controller.text.trim().isNotEmpty && !widget.isSending;
    return Container(
      padding: const EdgeInsets.fromLTRB(22, 12, 22, 18),
      decoration: const BoxDecoration(
        color: AppTheme.sidebar,
        border: Border(
          top: BorderSide(
            color: AppTheme.border,
            width: 1,
          ),
        ),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Expanded(
            child: Focus(
              onKeyEvent: (node, event) {
                if (event is KeyDownEvent &&
                    event.logicalKey == LogicalKeyboardKey.enter &&
                    !HardwareKeyboard.instance.isShiftPressed) {
                  _handleSend();
                  return KeyEventResult.handled;
                }

                return KeyEventResult.ignored;
              },
              child: TextField(
                controller: _controller,
                focusNode: _focusNode,
                onChanged: (_) => setState(() {}),
                minLines: 1,
                maxLines: 6,
                textInputAction: TextInputAction.send,
                onSubmitted: (_) {
                  if (!HardwareKeyboard.instance.isShiftPressed) {
                    _handleSend();
                  }
                },
                decoration: const InputDecoration(
                  hintText: "Ask SMARTA about a machine issue...",
                  prefixIcon: Icon(Icons.chat_bubble_outline, size: 19),
                  contentPadding:
                      EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                ),
              ),
            ),
          ),
          const SizedBox(width: 12),
          SizedBox(
            width: 48,
            height: 48,
            child: ElevatedButton(
              onPressed: canSend ? _handleSend : null,
              style: ElevatedButton.styleFrom(
                padding: EdgeInsets.zero,
                shape: const CircleBorder(),
              ),
              child: widget.isSending
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: Colors.white,
                      ),
                    )
                  : const Icon(
                      Icons.send_rounded,
                      size: 20,
                    ),
            ),
          ),
        ],
      ),
    );
  }
}
