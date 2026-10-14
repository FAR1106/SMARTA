import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../providers/chat_provider.dart';
import '../theme.dart';
import '../widgets/chat_area.dart';
import '../widgets/right_panel.dart';
import '../widgets/sidebar.dart';
import 'maintenance_history_screen.dart';
import 'overview_screen.dart';
import 'predict_screen.dart';

enum _WorkspaceSection { overview, assistant, intelligence, history }

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  _WorkspaceSection _section = _WorkspaceSection.overview;

  static const _labels = ['Overview', 'Assistant', 'Workbench', 'History'];
  static const _icons = [
    Icons.space_dashboard_outlined,
    Icons.forum_outlined,
    Icons.query_stats_outlined,
    Icons.history,
  ];

  void _select(int index) =>
      setState(() => _section = _WorkspaceSection.values[index]);

  Future<void> _showChatHistory() => showDialog<void>(
        context: context,
        builder: (context) => Dialog(
          clipBehavior: Clip.antiAlias,
          child: SizedBox(
            width: 360,
            height: 600,
            child:
                Sidebar(onSessionSelected: () => Navigator.of(context).pop()),
          ),
        ),
      );

  @override
  Widget build(BuildContext context) {
    final selectedIndex = _section.index;
    return ChangeNotifierProvider(
      create: (_) => ChatProvider(),
      child: LayoutBuilder(builder: (context, constraints) {
        final compact = constraints.maxWidth < 1050;
        final showSessionSidebar = constraints.maxWidth >= 1250;
        final showRightPanel = constraints.maxWidth >= 1380;
        return Scaffold(
          appBar: PreferredSize(
            preferredSize: Size.fromHeight(compact ? 64 : 76),
            child: _TopNavigation(
              compact: compact,
              activeIndex: selectedIndex,
              onSelect: _select,
              onShowHistory: _showChatHistory,
            ),
          ),
          body: Stack(
            children: [
              const Positioned.fill(child: _PaperGrid()),
              Positioned.fill(
                child: IndexedStack(
                  index: selectedIndex,
                  children: [
                    OverviewScreen(
                      onOpenMachineIntelligence: () =>
                          _select(_WorkspaceSection.intelligence.index),
                      onOpenAssistant: () =>
                          _select(_WorkspaceSection.assistant.index),
                    ),
                    _AssistantWorkspace(
                      showSessionSidebar: showSessionSidebar,
                      showRightPanel: showRightPanel,
                    ),
                    PredictScreen(
                      embedded: true,
                      onAskAboutPrediction: (predictionContext) {
                        context
                            .read<ChatProvider>()
                            .startPredictionChat(predictionContext);
                        _select(_WorkspaceSection.assistant.index);
                      },
                    ),
                    const MaintenanceHistoryScreen(),
                  ],
                ),
              ),
            ],
          ),
        );
      }),
    );
  }
}

class _TopNavigation extends StatelessWidget implements PreferredSizeWidget {
  final bool compact;
  final int activeIndex;
  final ValueChanged<int> onSelect;
  final VoidCallback onShowHistory;

  const _TopNavigation(
      {required this.compact,
      required this.activeIndex,
      required this.onSelect,
      required this.onShowHistory});

  @override
  Size get preferredSize => Size.fromHeight(compact ? 64 : 76);

  @override
  Widget build(BuildContext context) => Material(
        color: AppTheme.sidebar.withValues(alpha: .97),
        child: Container(
          decoration: const BoxDecoration(
              border: Border(bottom: BorderSide(color: AppTheme.border))),
          padding: EdgeInsets.symmetric(horizontal: compact ? 16 : 36),
          child: Row(
            children: [
              _Brand(compact: compact),
              const Spacer(),
              if (!compact) ...[
                for (var i = 0; i < _HomeScreenState._labels.length; i++)
                  _NavItem(
                      label: _HomeScreenState._labels[i],
                      selected: activeIndex == i,
                      onTap: () => onSelect(i)),
                const Spacer(),
              ] else
                PopupMenuButton<int>(
                  tooltip: 'Navigate SMARTA',
                  icon: const Icon(Icons.menu_rounded),
                  onSelected: onSelect,
                  itemBuilder: (context) => [
                    for (var i = 0; i < _HomeScreenState._labels.length; i++)
                      PopupMenuItem(
                          value: i,
                          child: Row(children: [
                            Icon(_HomeScreenState._icons[i], size: 19),
                            const SizedBox(width: 10),
                            Text(_HomeScreenState._labels[i])
                          ])),
                  ],
                ),
              Consumer<ChatProvider>(builder: (context, provider, _) {
                final connected = provider.serverStatus == 'Connected';
                return Tooltip(
                  message:
                      'API ${connected ? 'reachable' : provider.serverStatus.toLowerCase()}',
                  child: OutlinedButton.icon(
                    onPressed: provider.refreshServerStatus,
                    style: OutlinedButton.styleFrom(
                        padding: EdgeInsets.symmetric(
                            horizontal: compact ? 10 : 16, vertical: 10),
                        backgroundColor: AppTheme.card),
                    icon: Icon(Icons.circle,
                        size: 8,
                        color: connected ? AppTheme.success : AppTheme.warning),
                    label: Text(
                        compact
                            ? (connected ? 'API' : 'Checking')
                            : (connected ? 'API connected' : 'API checking'),
                        style: const TextStyle(fontSize: 12)),
                  ),
                );
              }),
              if (activeIndex == 1 && !compact) ...[
                const SizedBox(width: 8),
                IconButton(
                    tooltip: 'Chat history',
                    onPressed: onShowHistory,
                    icon: const Icon(Icons.manage_search)),
                IconButton(
                    tooltip: 'New chat',
                    onPressed: () => context.read<ChatProvider>().newChat(),
                    icon: const Icon(Icons.add_comment_outlined)),
              ],
            ],
          ),
        ),
      );
}

class _Brand extends StatelessWidget {
  final bool compact;
  const _Brand({required this.compact});

  @override
  Widget build(BuildContext context) => Row(children: [
        Container(
            width: 38,
            height: 38,
            decoration: BoxDecoration(
                color: AppTheme.textPrimary,
                borderRadius: BorderRadius.circular(12)),
            child: const Icon(Icons.bolt, color: AppTheme.card, size: 23)),
        const SizedBox(width: 11),
        Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text('SMARTA',
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.w800,
                      letterSpacing: .2,
                      height: 1)),
              if (!compact)
                Text('MAINTENANCE INTELLIGENCE',
                    style: Theme.of(context)
                        .textTheme
                        .labelSmall
                        ?.copyWith(fontSize: 9, letterSpacing: 1.2)),
            ]),
      ]);
}

class _NavItem extends StatelessWidget {
  final String label;
  final bool selected;
  final VoidCallback onTap;
  const _NavItem(
      {required this.label, required this.selected, required this.onTap});

  @override
  Widget build(BuildContext context) => Semantics(
        button: true,
        selected: selected,
        child: InkWell(
          onTap: onTap,
          child: Container(
            margin: const EdgeInsets.symmetric(horizontal: 5),
            padding: const EdgeInsets.symmetric(horizontal: 13),
            decoration: BoxDecoration(
                border: Border(
                    bottom: BorderSide(
                        color: selected ? AppTheme.accent : Colors.transparent,
                        width: 3))),
            alignment: Alignment.center,
            child: Text(label,
                style: Theme.of(context).textTheme.titleSmall?.copyWith(
                    color: selected ? AppTheme.accent : AppTheme.textPrimary,
                    fontWeight: selected ? FontWeight.w700 : FontWeight.w600)),
          ),
        ),
      );
}

class _PaperGrid extends StatelessWidget {
  const _PaperGrid();
  @override
  Widget build(BuildContext context) =>
      IgnorePointer(child: CustomPaint(painter: _GridPainter()));
}

class _GridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = AppTheme.textPrimary.withValues(alpha: .025)
      ..strokeWidth = 1;
    const gap = 56.0;
    for (double x = 0; x < size.width; x += gap) {
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), paint);
    }
    for (double y = 0; y < size.height; y += gap) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class _AssistantWorkspace extends StatelessWidget {
  final bool showSessionSidebar;
  final bool showRightPanel;
  const _AssistantWorkspace(
      {required this.showSessionSidebar, required this.showRightPanel});

  @override
  Widget build(BuildContext context) => Padding(
        padding:
            EdgeInsets.all(MediaQuery.sizeOf(context).width < 760 ? 8 : 22),
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 1720),
            child: Card(
              clipBehavior: Clip.antiAlias,
              child: Row(children: [
                if (showSessionSidebar) ...[
                  const SizedBox(width: 290, child: Sidebar()),
                  const VerticalDivider(width: 1)
                ],
                const Expanded(child: ChatArea()),
                if (showRightPanel) ...[
                  const VerticalDivider(width: 1),
                  const SizedBox(width: 320, child: RightPanel())
                ],
              ]),
            ),
          ),
        ),
      );
}
