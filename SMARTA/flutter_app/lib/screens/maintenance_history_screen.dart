import 'package:flutter/material.dart';

import '../services/api_service.dart';
import '../theme.dart';

class MaintenanceHistoryScreen extends StatefulWidget {
  const MaintenanceHistoryScreen({super.key});

  @override
  State<MaintenanceHistoryScreen> createState() =>
      _MaintenanceHistoryScreenState();
}

class _MaintenanceHistoryScreenState extends State<MaintenanceHistoryScreen> {
  final _search = TextEditingController();
  late Future<Map<String, dynamic>> _history;
  String _filter = '';

  @override
  void initState() {
    super.initState();
    _history = ApiService.getMaintenanceHistory(limit: 100);
    _search.addListener(
        () => setState(() => _filter = _search.text.trim().toLowerCase()));
  }

  @override
  void dispose() {
    _search.dispose();
    super.dispose();
  }

  void _retry() => setState(() {
        _history = ApiService.getMaintenanceHistory(limit: 100);
      });

  @override
  Widget build(BuildContext context) {
    final width = MediaQuery.sizeOf(context).width;
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 1120),
        child: ListView(
          padding: EdgeInsets.fromLTRB(
              width < 600 ? 16 : 28, 24, width < 600 ? 16 : 28, 32),
          children: [
            Text.rich(
              const TextSpan(children: [
                TextSpan(text: 'Maintenance '),
                TextSpan(
                  text: 'history',
                  style: TextStyle(
                    fontFamily: 'Georgia',
                    fontStyle: FontStyle.italic,
                    color: AppTheme.botanical,
                  ),
                ),
              ]),
              style: Theme.of(context).textTheme.headlineLarge,
            ),
            const SizedBox(height: 5),
            Text(
              'Records read from SMARTA’s local indexed source data. The source workbook has no event dates, sensor values, or verified repair outcomes.',
              style: Theme.of(context).textTheme.bodyMedium,
            ),
            const SizedBox(height: 18),
            TextField(
              controller: _search,
              decoration: InputDecoration(
                prefixIcon: const Icon(Icons.search),
                hintText: 'Search machine, fault, cause, or action',
                suffixIcon: _filter.isEmpty
                    ? null
                    : IconButton(
                        tooltip: 'Clear search',
                        onPressed: _search.clear,
                        icon: const Icon(Icons.close),
                      ),
              ),
            ),
            const SizedBox(height: 16),
            FutureBuilder<Map<String, dynamic>>(
              future: _history,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return const _HistoryStateCard(
                    icon: Icons.hourglass_top,
                    title: 'Loading local records',
                    detail: 'Reading the indexed maintenance history.',
                  );
                }
                if (snapshot.hasError) {
                  return _HistoryStateCard(
                    icon: Icons.error_outline,
                    title: 'Maintenance history unavailable',
                    detail: snapshot.error
                        .toString()
                        .replaceFirst('Exception: ', ''),
                    action: TextButton.icon(
                      onPressed: _retry,
                      icon: const Icon(Icons.refresh),
                      label: const Text('Retry'),
                    ),
                  );
                }
                final payload = snapshot.data!;
                final total =
                    payload['total'] is num ? payload['total'] as num : 0;
                final items = (payload['items'] as List<dynamic>? ?? const [])
                    .whereType<Map<String, dynamic>>()
                    .where(_matches)
                    .toList();
                if (items.isEmpty) {
                  return _HistoryStateCard(
                    icon: Icons.inbox_outlined,
                    title: _filter.isEmpty
                        ? 'No records in the local index'
                        : 'No matching records',
                    detail: _filter.isEmpty
                        ? 'Add maintenance source files and refresh the corpus to make records available.'
                        : 'Try a different machine ID or fault description.',
                  );
                }
                return Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Text('${items.length} shown',
                            style: Theme.of(context).textTheme.titleMedium),
                        const SizedBox(width: 8),
                        Text('of $total indexed records',
                            style: Theme.of(context).textTheme.bodySmall),
                      ],
                    ),
                    const SizedBox(height: 10),
                    ...items.map((item) => _MaintenanceCard(item: item)),
                  ],
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  bool _matches(Map<String, dynamic> item) {
    if (_filter.isEmpty) return true;
    return [
      item['machine'],
      item['fault'],
      item['symptoms'],
      item['cause'],
      item['action'],
      item['source'],
    ].any(
        (value) => value?.toString().toLowerCase().contains(_filter) ?? false);
  }
}

class _MaintenanceCard extends StatelessWidget {
  final Map<String, dynamic> item;

  const _MaintenanceCard({required this.item});

  @override
  Widget build(BuildContext context) {
    final source = item['source']?.toString() ?? 'Source not provided';
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Wrap(
              spacing: 8,
              runSpacing: 8,
              crossAxisAlignment: WrapCrossAlignment.center,
              children: [
                _Tag(
                    text:
                        item['machine']?.toString() ?? 'Machine not specified'),
                _Tag(text: source, subdued: true),
              ],
            ),
            const SizedBox(height: 12),
            Text(item['fault']?.toString() ?? 'Fault not specified',
                style: Theme.of(context).textTheme.titleLarge),
            if (_hasText(item['symptoms'])) ...[
              const SizedBox(height: 10),
              _Field(label: 'Symptoms', value: item['symptoms'].toString()),
            ],
            if (_hasText(item['cause'])) ...[
              const SizedBox(height: 10),
              _Field(label: 'Recorded cause', value: item['cause'].toString()),
            ],
            if (_hasText(item['action'])) ...[
              const SizedBox(height: 10),
              _Field(
                  label: 'Recorded action', value: item['action'].toString()),
            ],
            const SizedBox(height: 12),
            Row(
              children: [
                const Icon(Icons.event_busy_outlined,
                    size: 16, color: AppTheme.textSecondary),
                const SizedBox(width: 7),
                Expanded(
                  child: Text('Event date not present in source',
                      style: Theme.of(context).textTheme.bodySmall),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  bool _hasText(dynamic value) =>
      value != null && value.toString().trim().isNotEmpty;
}

class _Field extends StatelessWidget {
  final String label;
  final String value;

  const _Field({required this.label, required this.value});

  @override
  Widget build(BuildContext context) => Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label.toUpperCase(),
              style: Theme.of(context).textTheme.labelSmall?.copyWith(
                    color: AppTheme.textSecondary,
                    letterSpacing: .6,
                  )),
          const SizedBox(height: 3),
          Text(value),
        ],
      );
}

class _Tag extends StatelessWidget {
  final String text;
  final bool subdued;

  const _Tag({required this.text, this.subdued = false});

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: subdued
              ? AppTheme.card2.withValues(alpha: .6)
              : AppTheme.accent.withValues(alpha: .18),
          borderRadius: BorderRadius.circular(20),
        ),
        child: Text(text,
            style: Theme.of(context).textTheme.labelSmall?.copyWith(
                  color: subdued ? AppTheme.textSecondary : AppTheme.accent,
                )),
      );
}

class _HistoryStateCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String detail;
  final Widget? action;

  const _HistoryStateCard({
    required this.icon,
    required this.title,
    required this.detail,
    this.action,
  });

  @override
  Widget build(BuildContext context) => Card(
        child: Padding(
          padding: const EdgeInsets.all(18),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(icon, color: AppTheme.textSecondary),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title, style: Theme.of(context).textTheme.titleMedium),
                    const SizedBox(height: 4),
                    Text(detail, style: Theme.of(context).textTheme.bodyMedium),
                    if (action != null) action!,
                  ],
                ),
              ),
            ],
          ),
        ),
      );
}
