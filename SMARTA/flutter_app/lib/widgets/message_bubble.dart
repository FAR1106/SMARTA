import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';

import '../theme.dart';

class MessageBubble extends StatelessWidget {
  final String text;
  final bool isUser;
  final List<String> sources;
  final bool failed;
  final String? errorCode;
  final VoidCallback? onRetry;

  const MessageBubble({
    super.key,
    required this.text,
    required this.isUser,
    this.sources = const [],
    this.failed = false,
    this.errorCode,
    this.onRetry,
  });

  @override
  Widget build(BuildContext context) {
    final availableWidth = MediaQuery.sizeOf(context).width;
    final maxWidth = math.min(840.0, availableWidth * .88);
    final bodyStyle = Theme.of(context).textTheme.bodyMedium!.copyWith(
          color: AppTheme.textPrimary,
          fontSize: 14,
          height: 1.55,
        );

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: ConstrainedBox(
        constraints: BoxConstraints(maxWidth: maxWidth),
        child: Container(
          margin: const EdgeInsets.symmetric(vertical: 8),
          padding:
              EdgeInsets.symmetric(horizontal: isUser ? 18 : 18, vertical: 14),
          decoration: isUser
              ? BoxDecoration(
                  color: AppTheme.accent,
                  borderRadius: BorderRadius.circular(18),
                )
              : failed
                  ? BoxDecoration(
                      color: AppTheme.dangerSurface,
                      borderRadius: BorderRadius.circular(18),
                      border: Border.all(
                          color: AppTheme.danger.withValues(alpha: .3)),
                    )
                  : BoxDecoration(
                      color: AppTheme.card,
                      borderRadius: BorderRadius.circular(18),
                      border: Border.all(color: AppTheme.border),
                    ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              if (failed) ...[
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(Icons.error_outline,
                        color: AppTheme.danger, size: 17),
                    const SizedBox(width: 7),
                    Text(_failureTitle(errorCode),
                        style: Theme.of(context).textTheme.labelLarge?.copyWith(
                              color: AppTheme.danger,
                            )),
                  ],
                ),
                const SizedBox(height: 6),
                if (onRetry != null)
                  Align(
                    alignment: Alignment.centerLeft,
                    child: TextButton.icon(
                      onPressed: onRetry,
                      icon: const Icon(Icons.refresh, size: 16),
                      label: const Text('Retry'),
                    ),
                  ),
              ],
              if (isUser)
                SelectableText(
                  text,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 14,
                    height: 1.5,
                  ),
                )
              else
                MarkdownBody(
                  data: text,
                  selectable: true,
                  styleSheet: MarkdownStyleSheet(
                    p: bodyStyle,
                    pPadding: const EdgeInsets.only(bottom: 9),
                    h1: bodyStyle.copyWith(
                        fontSize: 21, fontWeight: FontWeight.w700),
                    h1Padding: const EdgeInsets.only(top: 8, bottom: 5),
                    h2: bodyStyle.copyWith(
                        fontSize: 18, fontWeight: FontWeight.w700),
                    h2Padding: const EdgeInsets.only(top: 8, bottom: 4),
                    h3: bodyStyle.copyWith(
                        fontSize: 16, fontWeight: FontWeight.w700),
                    h3Padding: const EdgeInsets.only(top: 6, bottom: 3),
                    h4: bodyStyle.copyWith(fontWeight: FontWeight.w700),
                    h5: bodyStyle.copyWith(fontWeight: FontWeight.w700),
                    h6: bodyStyle.copyWith(fontWeight: FontWeight.w700),
                    a: bodyStyle.copyWith(
                      color: AppTheme.accent,
                      decoration: TextDecoration.underline,
                    ),
                    em: bodyStyle.copyWith(fontStyle: FontStyle.italic),
                    strong: bodyStyle.copyWith(fontWeight: FontWeight.w700),
                    code: bodyStyle.copyWith(
                      fontFamily: 'monospace',
                      fontSize: 13,
                      backgroundColor: AppTheme.card2,
                    ),
                    blockquote: bodyStyle.copyWith(
                      color: AppTheme.textSecondary,
                      fontStyle: FontStyle.italic,
                    ),
                    blockquotePadding:
                        const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                    blockquoteDecoration: const BoxDecoration(
                      border: Border(
                        left: BorderSide(color: AppTheme.accent, width: 3),
                      ),
                    ),
                    codeblockPadding: const EdgeInsets.all(12),
                    codeblockDecoration: BoxDecoration(
                        color: AppTheme.card2,
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(color: AppTheme.border)),
                    listIndent: 24,
                    listBullet: bodyStyle,
                    tableHead: bodyStyle.copyWith(fontWeight: FontWeight.w700),
                    tableBody: bodyStyle,
                    tableBorder: TableBorder.all(color: AppTheme.border),
                    tableCellsPadding: const EdgeInsets.all(7),
                    blockSpacing: 7,
                  ),
                ),
              if (!isUser && sources.isNotEmpty) ...[
                const SizedBox(height: 10),
                const Divider(height: 1, color: AppTheme.border),
                const SizedBox(height: 8),
                Text('Sources',
                    style: Theme.of(context).textTheme.labelMedium?.copyWith(
                          color: AppTheme.accent,
                          fontWeight: FontWeight.w700,
                        )),
                const SizedBox(height: 4),
                for (final source in sources)
                  Padding(
                    padding: const EdgeInsets.only(top: 3),
                    child: Text(source,
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                              color: AppTheme.textSecondary,
                              height: 1.35,
                            )),
                  ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  String _failureTitle(String? code) => switch (code) {
        'language_model_unavailable' => 'Language model unavailable',
        'generation_timeout' => 'Language model timed out',
        'invalid_response' => 'Invalid response',
        'retrieval_failed' => 'Evidence search failed',
        'api_request_timeout' => 'API request timed out',
        _ => 'Request failed',
      };
}
