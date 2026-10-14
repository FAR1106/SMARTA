import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/chat_session.dart';
import '../providers/chat_provider.dart';
import '../theme.dart';

class Sidebar extends StatefulWidget {
  final VoidCallback? onSessionSelected;
  const Sidebar({super.key, this.onSessionSelected});

  @override
  State<Sidebar> createState() => _SidebarState();
}

class _SidebarState extends State<Sidebar> {
  final _search = TextEditingController();
  String _query = '';

  @override
  void initState() {
    super.initState();
    _search.addListener(
        () => setState(() => _query = _search.text.trim().toLowerCase()));
  }

  @override
  void dispose() {
    _search.dispose();
    _searchFocus.dispose();
    super.dispose();
  }

  Future<void> _renameDialog(
      BuildContext context, ChatProvider provider, ChatSession session) async {
    final controller = TextEditingController(text: session.title);
    final newTitle = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: AppTheme.card,
        title: const Text('Rename chat'),
        content: TextField(
            controller: controller,
            autofocus: true,
            decoration: const InputDecoration(hintText: 'Chat title')),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancel')),
          ElevatedButton(
              onPressed: () => Navigator.pop(context, controller.text.trim()),
              child: const Text('Save'))
        ],
      ),
    );
    controller.dispose();
    if (newTitle != null && newTitle.isNotEmpty) {
      provider.renameChat(session.id, newTitle);
    }
  }

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<ChatProvider>();
    final sessions = provider.sessions
        .where((session) =>
            _query.isEmpty ||
            session.title.toLowerCase().contains(_query) ||
            session.messages
                .any((message) => message.text.toLowerCase().contains(_query)))
        .toList();
    final activeId = provider.activeSession.id;
    return Container(
      color: AppTheme.card,
      child: Column(children: [
        Padding(
            padding: const EdgeInsets.fromLTRB(16, 17, 12, 12),
            child: Row(children: [
              Container(
                  width: 32,
                  height: 32,
                  decoration: BoxDecoration(
                      color: AppTheme.textPrimary,
                      borderRadius: BorderRadius.circular(10)),
                  child: const Icon(Icons.forum_outlined,
                      color: AppTheme.card, size: 18)),
              const SizedBox(width: 9),
              Expanded(
                  child: Text('Chat history',
                      style: Theme.of(context).textTheme.titleLarge)),
              IconButton(
                  tooltip: 'Search chats',
                  onPressed: () =>
                      FocusScope.of(context).requestFocus(_searchFocus),
                  icon: const Icon(Icons.search, size: 20)),
            ])),
        Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            child: TextField(
                controller: _search,
                focusNode: _searchFocus,
                decoration: InputDecoration(
                    hintText: 'Search conversations',
                    prefixIcon: const Icon(Icons.search, size: 18),
                    suffixIcon: _query.isEmpty
                        ? null
                        : IconButton(
                            onPressed: _search.clear,
                            icon: const Icon(Icons.close, size: 18)),
                    contentPadding: const EdgeInsets.symmetric(
                        vertical: 10, horizontal: 12)),
                style: Theme.of(context).textTheme.bodySmall)),
        Padding(
          padding: const EdgeInsets.fromLTRB(12, 10, 12, 8),
          child: SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                  onPressed: provider.newChat,
                  icon: const Icon(Icons.add, size: 18),
                  label: const Text('New chat'))),
        ),
        const Divider(height: 1),
        Expanded(
            child: sessions.isEmpty
                ? Center(
                    child: Padding(
                        padding: const EdgeInsets.all(18),
                        child: Text(
                            _query.isEmpty
                                ? 'No conversations yet.'
                                : 'No matching conversations.',
                            style: Theme.of(context).textTheme.bodySmall,
                            textAlign: TextAlign.center)))
                : ListView.builder(
                    padding: const EdgeInsets.symmetric(vertical: 8),
                    itemCount: sessions.length,
                    itemBuilder: (context, index) {
                      final session = sessions[index];
                      final isActive = session.id == activeId;
                      final time = session.messages.isEmpty
                          ? 'New conversation'
                          : _timeLabel(session.messages.last.timestamp);
                      return Container(
                        margin: const EdgeInsets.symmetric(
                            horizontal: 8, vertical: 3),
                        decoration: BoxDecoration(
                            color: isActive
                                ? AppTheme.accentSoft
                                : Colors.transparent,
                            borderRadius: BorderRadius.circular(14)),
                        child: Material(
                          color: Colors.transparent,
                          borderRadius: BorderRadius.circular(14),
                          clipBehavior: Clip.antiAlias,
                          child: ListTile(
                            dense: true,
                            leading: Icon(
                                session.predictionContext == null
                                    ? Icons.forum_outlined
                                    : Icons.analytics_outlined,
                                size: 19,
                                color: isActive
                                    ? AppTheme.accent
                                    : AppTheme.textSecondary),
                            title: Text(session.title,
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                                style: Theme.of(context)
                                    .textTheme
                                    .titleSmall
                                    ?.copyWith(fontSize: 13)),
                            subtitle: Text(time,
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                                style: Theme.of(context)
                                    .textTheme
                                    .bodySmall
                                    ?.copyWith(fontSize: 11)),
                            onTap: () {
                              provider.selectChat(session.id);
                              widget.onSessionSelected?.call();
                            },
                            trailing: PopupMenuButton<String>(
                                color: AppTheme.card,
                                icon: const Icon(Icons.more_horiz, size: 18),
                                onSelected: (value) {
                                  if (value == 'rename') {
                                    _renameDialog(context, provider, session);
                                  }
                                  if (value == 'delete') {
                                    provider.deleteChat(session.id);
                                  }
                                },
                                itemBuilder: (context) => const [
                                      PopupMenuItem(
                                          value: 'rename',
                                          child: Text('Rename')),
                                      PopupMenuItem(
                                          value: 'delete',
                                          child: Text('Delete'))
                                    ]),
                          ),
                        ),
                      );
                    })),
      ]),
    );
  }

  final FocusNode _searchFocus = FocusNode();

  String _timeLabel(DateTime time) {
    final now = DateTime.now();
    if (now.difference(time).inDays == 0) {
      final hour = time.hour % 12 == 0 ? 12 : time.hour % 12;
      final minute = time.minute.toString().padLeft(2, '0');
      return 'Today, $hour:$minute ${time.hour >= 12 ? 'PM' : 'AM'}';
    }
    return '${time.month}/${time.day}/${time.year}';
  }
}
