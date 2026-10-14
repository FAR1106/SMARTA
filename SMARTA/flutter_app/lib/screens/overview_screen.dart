import 'package:flutter/material.dart';

import '../theme.dart';

class OverviewScreen extends StatelessWidget {
  final VoidCallback onOpenMachineIntelligence;
  final VoidCallback onOpenAssistant;

  const OverviewScreen({
    super.key,
    required this.onOpenMachineIntelligence,
    required this.onOpenAssistant,
  });

  @override
  Widget build(BuildContext context) {
    final width = MediaQuery.sizeOf(context).width;
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 1320),
        child: ListView(
          padding: EdgeInsets.fromLTRB(
              width < 600 ? 16 : 28, 24, width < 600 ? 16 : 28, 32),
          children: [
            _OverviewHero(
              onAsk: onOpenAssistant,
              onWorkbench: onOpenMachineIntelligence,
            ),
          ],
        ),
      ),
    );
  }
}

class _OverviewHero extends StatelessWidget {
  final VoidCallback onAsk;
  final VoidCallback onWorkbench;

  const _OverviewHero({required this.onAsk, required this.onWorkbench});

  @override
  Widget build(BuildContext context) {
    final compact = MediaQuery.sizeOf(context).width < 700;
    final headlineStyle = Theme.of(context)
        .textTheme
        .displayMedium
        ?.copyWith(fontSize: compact ? 38 : 54, height: 1.02);
    return Padding(
      padding: EdgeInsets.fromLTRB(8, compact ? 20 : 46, 8, compact ? 24 : 42),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Text.rich(
            const TextSpan(children: [
              TextSpan(text: 'Stop '),
              TextSpan(
                  text: 'guessing.',
                  style: TextStyle(
                      fontFamily: 'Georgia',
                      fontStyle: FontStyle.italic,
                      fontWeight: FontWeight.w700,
                      color: AppTheme.danger,
                      decoration: TextDecoration.lineThrough,
                      decorationThickness: 2)),
              TextSpan(text: '\nAsk SMARTA for the\n'),
              TextSpan(
                  text: 'evidence.',
                  style: TextStyle(
                      fontFamily: 'Georgia',
                      fontStyle: FontStyle.italic,
                      fontWeight: FontWeight.w700,
                      color: AppTheme.botanical)),
            ]),
            textAlign: TextAlign.center,
            style: headlineStyle,
          ),
          const SizedBox(height: 18),
          ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 720),
            child: Text(
              'SMARTA is a maintenance assistant for your machines. Search recorded faults, causes and actions, and classify supplied operating points. Every answer shows where its evidence came from.',
              textAlign: TextAlign.center,
              style:
                  Theme.of(context).textTheme.bodyLarge?.copyWith(fontSize: 17),
            ),
          ),
          const SizedBox(height: 16),
          Text.rich(
            const TextSpan(children: [
              TextSpan(text: 'Fault on the line? '),
              TextSpan(
                  text: 'Ask SMARTA',
                  style: TextStyle(
                      color: AppTheme.accent, fontWeight: FontWeight.w700)),
              TextSpan(text: ' — answers '),
              TextSpan(
                  text: 'come with sources.',
                  style: TextStyle(
                      color: AppTheme.botanical, fontWeight: FontWeight.w700)),
            ]),
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 23),
          Wrap(
              alignment: WrapAlignment.center,
              spacing: 12,
              runSpacing: 10,
              children: [
                FilledButton.icon(
                    onPressed: onAsk,
                    icon: const Icon(Icons.arrow_forward_rounded),
                    label: const Text('Ask SMARTA')),
                OutlinedButton.icon(
                    onPressed: onWorkbench,
                    icon: const Icon(Icons.tune_rounded),
                    label: const Text('Open the workbench')),
              ]),
        ],
      ),
    );
  }
}
