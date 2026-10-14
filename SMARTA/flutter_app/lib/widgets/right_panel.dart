import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../providers/chat_provider.dart';
import '../services/api_service.dart';
import '../theme.dart';

class RightPanel extends StatefulWidget {
  const RightPanel({super.key});

  @override
  State<RightPanel> createState() => _RightPanelState();
}

class _RightPanelState extends State<RightPanel> {
  late Future<Map<String, dynamic>> _health;

  @override
  void initState() {
    super.initState();
    _health = ApiService.getSystemHealth();
  }

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<ChatProvider>();
    final connected = provider.serverStatus == 'Connected';
    final assistantMessages =
        provider.messages.reversed.where((message) => !message.isUser).toList();
    final latestAssistant =
        assistantMessages.isEmpty ? null : assistantMessages.first;
    final latestAnswer =
        latestAssistant?.failed == true ? null : latestAssistant;
    return Container(
      color: AppTheme.sidebar.withValues(alpha: .5),
      padding: const EdgeInsets.all(18),
      child: ListView(
        children: [
          Text('Evidence used',
              style: Theme.of(context)
                  .textTheme
                  .headlineSmall
                  ?.copyWith(fontSize: 20)),
          const SizedBox(height: 12),
          if (latestAssistant?.failed == true)
            _PanelCard(
                child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                  const Icon(Icons.info_outline, color: AppTheme.warning),
                  const SizedBox(height: 10),
                  Text('Evidence status unavailable',
                      style: Theme.of(context).textTheme.titleSmall),
                  const SizedBox(height: 5),
                  Text(
                      'This request failed before a cited answer was returned. Its source status is unknown; retry after resolving the request error.',
                      style: Theme.of(context).textTheme.bodySmall),
                ]))
          else if (latestAnswer == null || latestAnswer.sources.isEmpty)
            _PanelCard(
                child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                  const Icon(Icons.menu_book_outlined, color: AppTheme.accent),
                  const SizedBox(height: 10),
                  Text(
                      latestAnswer?.evidenceStatus == 'no_relevant_evidence'
                          ? 'No relevant evidence found'
                          : 'No cited sources returned',
                      style: Theme.of(context).textTheme.titleSmall),
                  const SizedBox(height: 5),
                  Text(
                      latestAnswer?.evidenceStatus == 'no_relevant_evidence'
                          ? 'SMARTA searched the indexed records but found no relevant passages for this question.'
                          : 'SMARTA returned no source references with this answer.',
                      style: Theme.of(context).textTheme.bodySmall),
                ]))
          else
            _PanelCard(
                child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                  for (var i = 0; i < latestAnswer.sources.length; i++) ...[
                    if (i > 0) const Divider(height: 20),
                    Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Icon(Icons.description_outlined,
                              size: 17, color: AppTheme.accent),
                          const SizedBox(width: 8),
                          Expanded(
                              child: SelectableText(latestAnswer.sources[i],
                                  style: Theme.of(context)
                                      .textTheme
                                      .bodySmall
                                      ?.copyWith(
                                          color: AppTheme.textPrimary,
                                          height: 1.4))),
                        ]),
                  ],
                  const SizedBox(height: 12),
                  Text(
                      'These are the source references returned for the selected answer. Full passages are shown only when the response provides them.',
                      style: Theme.of(context).textTheme.bodySmall),
                ])),
          const SizedBox(height: 22),
          Text('Session',
              style: Theme.of(context)
                  .textTheme
                  .headlineSmall
                  ?.copyWith(fontSize: 20)),
          const SizedBox(height: 12),
          _PanelCard(
              child: Column(children: [
            _InfoRow(
                label: 'Status',
                value: provider.serverStatus,
                valueColor: connected ? AppTheme.success : AppTheme.warning),
            const SizedBox(height: 12),
            _InfoRow(label: 'Messages', value: '${provider.messages.length}'),
            const SizedBox(height: 12),
            _InfoRow(
                label: 'Conversations', value: '${provider.sessions.length}'),
          ])),
          const SizedBox(height: 12),
          FutureBuilder<Map<String, dynamic>>(
            future: _health,
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.waiting) {
                return const _PanelCard(
                    child: LinearProgressIndicator(minHeight: 2));
              }
              if (snapshot.hasError) {
                return const _PanelCard(
                    child: Text('Model and corpus details are unavailable.'));
              }
              final payload = snapshot.data!;
              final corpus = payload['corpus'] is Map<String, dynamic>
                  ? payload['corpus'] as Map<String, dynamic>
                  : const <String, dynamic>{};
              return _PanelCard(
                  child: Column(children: [
                _InfoRow(
                    label: 'Language model',
                    value:
                        payload['ollama_model']?.toString() ?? 'Not reported'),
                const SizedBox(height: 12),
                _InfoRow(
                    label: 'Language model status',
                    value: payload['ollama_status']?.toString() ?? 'Unknown',
                    valueColor: payload['ollama_status'] == 'ready'
                        ? AppTheme.success
                        : AppTheme.warning),
                const SizedBox(height: 12),
                _InfoRow(
                    label: 'Indexed passages',
                    value: corpus['documents']?.toString() ?? 'Unavailable'),
                const SizedBox(height: 12),
                _InfoRow(
                    label: 'Maintenance records',
                    value: corpus['maintenance_records']?.toString() ??
                        'Unavailable'),
              ]));
            },
          ),
        ],
      ),
    );
  }
}

class _PanelCard extends StatelessWidget {
  final Widget child;
  const _PanelCard({required this.child});
  @override
  Widget build(BuildContext context) =>
      Card(child: Padding(padding: const EdgeInsets.all(16), child: child));
}

class _InfoRow extends StatelessWidget {
  final String label;
  final String value;
  final Color? valueColor;
  const _InfoRow({required this.label, required this.value, this.valueColor});
  @override
  Widget build(BuildContext context) => Row(children: [
        Expanded(
            child: Text(label, style: Theme.of(context).textTheme.bodySmall)),
        const SizedBox(width: 8),
        Flexible(
            child: Text(value,
                textAlign: TextAlign.right,
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: valueColor ?? AppTheme.textPrimary,
                    fontWeight: FontWeight.w600))),
      ]);
}
