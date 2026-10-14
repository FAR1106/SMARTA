import 'dart:convert';
import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import 'package:provider/provider.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:smarta_chat/main.dart';
import 'package:smarta_chat/models/prediction_chat_context.dart';
import 'package:smarta_chat/models/chat_message.dart';
import 'package:smarta_chat/screens/predict_screen.dart';
import 'package:smarta_chat/providers/chat_provider.dart';
import 'package:smarta_chat/services/api_service.dart';
import 'package:smarta_chat/widgets/chat_area.dart';
import 'package:smarta_chat/widgets/message_bubble.dart';
import 'package:smarta_chat/widgets/right_panel.dart';

Finder _markdownContaining(String fragment) => find.byWidgetPredicate(
      (widget) => widget is MarkdownBody && widget.data.contains(fragment),
    );

Map<String, dynamic> _metadata() => {
      'ready': true,
      'model_version': 'ai4i-rf-v1',
      'trained_at': '2026-10-13T09:00:00Z',
      'allowed_types': ['L', 'M', 'H'],
      'supported_failure_modes': ['TWF', 'HDF', 'PWF', 'OSF'],
      'unsupported_failure_modes': {
        'RNF': 'Only 19 positive examples; minimum is 20.'
      },
      'feature_ranges': {
        'Air temperature [K]': [295.3, 304.5],
        'Process temperature [K]': [305.7, 313.8],
        'Rotational speed [rpm]': [1168.0, 2886.0],
        'Torque [Nm]': [3.8, 76.6],
        'Tool wear [min]': [0.0, 253.0],
      },
    };

Map<String, dynamic> _predictionResponse() => {
      'prediction': {
        'machine_id': 'Machine-01',
        'prediction_timestamp': '2026-10-13T09:00:00Z',
        'model_version': 'ai4i-rf-v1',
        'machine_failure': {
          'classification': 1,
          'score': 0.78,
          'score_type': 'uncalibrated_random_forest_score',
        },
        'failure_modes': {
          'TWF': {
            'classification': 0,
            'score': 0.49,
            'score_type': 'uncalibrated_random_forest_score',
          },
          'HDF': {
            'classification': 1,
            'score': 0.52,
            'score_type': 'uncalibrated_random_forest_score',
          },
        },
        'failure_mode_labels': ['heat dissipation failure'],
        'unsupported_failure_modes': {
          'RNF': 'Only 19 positive examples; minimum is 20.',
        },
        'input_features': {
          'Type': 'M',
          'Air temperature [K]': 298.1,
          'Process temperature [K]': 308.6,
          'Rotational speed [rpm]': 1551.0,
          'Torque [Nm]': 42.8,
          'Tool wear [min]': 20.0,
        },
        'out_of_training_range': {},
        'limitations':
            'AI4I synthetic-data classification only; not a future failure date.',
      },
      'evidence': [],
      'machine_specific_evidence': [],
      'recommendation':
          'No sufficiently supported corrective action was found in the available maintenance records or documentation. Further inspection is required.',
      'evidence_status': 'insufficient',
    };

void _setDesktopViewport(WidgetTester tester) {
  tester.view.physicalSize = const Size(1440, 900);
  tester.view.devicePixelRatio = 1;
  addTearDown(tester.view.resetPhysicalSize);
  addTearDown(tester.view.resetDevicePixelRatio);
}

Map<String, dynamic> _health() => {
      'status': 'ok',
      'corpus': {'documents': 1360, 'maintenance_records': 20, 'sources': 4},
      'mysql_credentials_configured': false,
      'ollama_model': 'qwen2.5:3b',
      'ollama_status': 'ready',
      'sensor_status': 'not_connected',
      'predictive_model': 'ready',
    };

Map<String, dynamic> _maintenanceHistory() => {
      'total': 1,
      'limit': 4,
      'offset': 0,
      'items': [
        {
          'id': 1,
          'machine': 'SMT13',
          'fault': 'Vacuum pressure low',
          'symptoms': 'Pickup failures',
          'cause': 'Blocked filter',
          'action': 'Cleaned filter',
          'source': 'maintenance.xlsx',
        },
      ],
    };

void main() {
  test('prediction chat context rejects missing and malformed API fields', () {
    expect(
      () => PredictionChatContext.fromPredictionResponse({}),
      throwsFormatException,
    );
    final malformed =
        jsonDecode(jsonEncode(_predictionResponse())) as Map<String, dynamic>;
    (malformed['prediction'] as Map<String, dynamic>)
        .remove('prediction_timestamp');
    expect(
      () => PredictionChatContext.fromPredictionResponse(malformed),
      throwsFormatException,
    );
  });

  test('prediction context remains scoped to its chat session', () async {
    final bodies = <Map<String, dynamic>>[];
    ApiService.client = MockClient((request) async {
      if (request.url.path.endsWith('/chat')) {
        bodies.add(jsonDecode(request.body) as Map<String, dynamic>);
        return http.Response(jsonEncode({'answer': 'ok', 'sources': []}), 200);
      }
      if (request.url.path.endsWith('/health')) {
        return http.Response(jsonEncode(_health()), 200);
      }
      return http.Response('{}', 404);
    });
    addTearDown(ApiService.client.close);
    final provider = ChatProvider();
    final attached = PredictionChatContext.fromPredictionResponse(
      jsonDecode(jsonEncode(_predictionResponse())) as Map<String, dynamic>,
    );

    provider.startPredictionChat(attached);
    await provider.sendMessage('Ask about this result');
    await provider.sendMessage('What inputs did the classifier use?');
    final firstPredictionSessionId = provider.activeSession.id;

    final secondResponse =
        jsonDecode(jsonEncode(_predictionResponse())) as Map<String, dynamic>;
    final secondPrediction =
        secondResponse['prediction'] as Map<String, dynamic>;
    secondPrediction['machine_id'] = 'Machine-02';
    secondPrediction['prediction_timestamp'] = '2026-10-13T10:00:00Z';
    final secondContext = PredictionChatContext.fromPredictionResponse(
      secondResponse,
    );
    provider.startPredictionChat(secondContext);
    await provider.sendMessage('Explain this other result');
    final secondPredictionSessionId = provider.activeSession.id;

    provider.selectChat(firstPredictionSessionId);
    await provider.sendMessage('Return to the first result');
    expect(provider.activeSession.predictionContext?.machineId, 'Machine-01');

    provider.newChat();
    expect(provider.activeSession.predictionContext, isNull);
    await provider.sendMessage('Ordinary documentation question');

    expect(bodies[0]['prediction_context']['machine_id'], 'Machine-01');
    expect(bodies[1]['prediction_context']['machine_id'], 'Machine-01');
    expect(bodies[2]['prediction_context']['machine_id'], 'Machine-02');
    expect(bodies[3]['prediction_context']['machine_id'], 'Machine-01');
    expect(bodies[4].containsKey('prediction_context'), isFalse);
    provider.selectChat(secondPredictionSessionId);
    expect(provider.activeSession.predictionContext?.machineId, 'Machine-02');
    provider.selectChat(firstPredictionSessionId);
    expect(provider.activeSession.predictionContext?.machineId, 'Machine-01');
    provider.dispose();
  });

  testWidgets('main navigation removes Fleet Simulator and keeps the workbench',
      (WidgetTester tester) async {
    _setDesktopViewport(tester);
    ApiService.client = MockClient((request) async {
      if (request.url.path.endsWith('/predict/metadata')) {
        return http.Response(jsonEncode(_metadata()), 200);
      }
      if (request.url.path.endsWith('/health')) {
        return http.Response(jsonEncode(_health()), 200);
      }
      if (request.url.path.endsWith('/maintenance/history')) {
        return http.Response(jsonEncode(_maintenanceHistory()), 200);
      }
      return http.Response('{}', 404);
    });
    addTearDown(ApiService.client.close);

    await tester.pumpWidget(const SmartaApp());
    await tester.pumpAndSettle();

    expect(find.text('Fleet simulator'), findsNothing);
    expect(find.text('Fleet'), findsNothing);
    expect(find.text('Workbench'), findsOneWidget);
    await tester.tap(find.text('Workbench'));
    await tester.pumpAndSettle();
    expect(find.text('Predictions dashboard'), findsOneWidget);
    expect(find.text('Machine Simulator'), findsOneWidget);
  });

  test('insufficient evidence and an API failure have distinct outcomes',
      () async {
    ApiService.client = MockClient((request) async => http.Response(
          jsonEncode({
            'answer':
                'Insufficient evidence in the available maintenance records.',
            'sources': [],
          }),
          200,
        ));
    addTearDown(ApiService.client.close);
    final insufficient =
        await ApiService.sendMessage('What caused this fault?');
    expect(insufficient.failed, isFalse);
    expect(insufficient.answer, contains('Insufficient evidence'));

    ApiService.client = MockClient((request) async => http.Response(
          '{"detail":"SMARTA could not generate a response."}',
          500,
        ));
    addTearDown(ApiService.client.close);
    final failed = await ApiService.sendMessage('What caused this fault?');
    expect(failed.failed, isTrue);
    expect(failed.answer, contains('could not generate'));
  });

  test('structured model unavailability is shown without provider details',
      () async {
    ApiService.client = MockClient((request) async => http.Response(
          jsonEncode({
            'detail': {
              'code': 'language_model_unavailable',
              'message':
                  'SMARTA\'s language model is unavailable. Start Ollama and retry.',
            }
          }),
          503,
        ));
    addTearDown(ApiService.client.close);
    final reply =
        await ApiService.sendMessage('What checks can prevent faults?');
    expect(reply.failed, isTrue);
    expect(reply.errorCode, 'language_model_unavailable');
    expect(reply.answer, contains('Start Ollama'));
  });

  test('retry reuses the failed question and preserves its prediction context',
      () async {
    final requestBodies = <Map<String, dynamic>>[];
    var attempts = 0;
    ApiService.client = MockClient((request) async {
      requestBodies.add(jsonDecode(request.body) as Map<String, dynamic>);
      attempts++;
      if (attempts == 1) {
        return http.Response(
            jsonEncode({
              'detail': {
                'code': 'language_model_unavailable',
                'message': 'Start Ollama and retry.',
              }
            }),
            503);
      }
      return http.Response(
          jsonEncode({
            'answer': 'Inspect the cooling circuit [1].',
            'sources': ['[1] manual.pdf, page 4'],
          }),
          200);
    });
    addTearDown(ApiService.client.close);
    final provider = ChatProvider();
    provider.startPredictionChat(PredictionChatContext.fromPredictionResponse(
      jsonDecode(jsonEncode(_predictionResponse())) as Map<String, dynamic>,
    ));
    await provider.sendMessage('Explain this machine prediction');
    expect(provider.messages.where((message) => message.isUser).length, 1);
    final failedReply = provider.messages.last;
    expect(failedReply.failed, isTrue);
    await provider.retryFailedMessage(failedReply);

    expect(provider.messages.where((message) => message.isUser).length, 1);
    expect(provider.messages.where((message) => !message.isUser).length, 1);
    expect(provider.messages.last.text, contains('cooling circuit'));
    expect(requestBodies.map((body) => body['question']).toList(),
        ['Explain this machine prediction', 'Explain this machine prediction']);
    expect(
        requestBodies.every((body) =>
            (body['prediction_context'] as Map)['machine_id'] == 'Machine-01'),
        isTrue);
    provider.dispose();
  });

  testWidgets('retry button resends the question without duplicating it',
      (WidgetTester tester) async {
    _setDesktopViewport(tester);
    final sentQuestions = <String>[];
    var attempts = 0;
    ApiService.client = MockClient((request) async {
      if (request.url.path.endsWith('/chat')) {
        final body = jsonDecode(request.body) as Map<String, dynamic>;
        sentQuestions.add(body['question'] as String);
        attempts++;
        if (attempts == 1) {
          return http.Response(
              jsonEncode({
                'detail': {
                  'code': 'language_model_unavailable',
                  'message': 'Start Ollama and retry.',
                }
              }),
              503);
        }
        return http.Response(
            jsonEncode({'answer': 'Recovered.', 'sources': []}), 200);
      }
      return http.Response(jsonEncode(_health()), 200);
    });
    addTearDown(ApiService.client.close);

    final provider = ChatProvider();
    await tester.pumpWidget(ChangeNotifierProvider.value(
        value: provider,
        child: const MaterialApp(home: Scaffold(body: ChatArea()))));
    await tester.pumpAndSettle();
    await tester.enterText(find.byType(TextField), 'compressor checks');
    await tester.pump();
    await tester.tap(find.byIcon(Icons.send_rounded));
    await tester.pumpAndSettle();
    expect(provider.messages.last.failed, isTrue);
    await tester.drag(find.byType(ListView).first, const Offset(0, -500));
    await tester.pumpAndSettle();
    expect(find.text('Retry'), findsOneWidget);
    await tester.tap(find.text('Retry'));
    await tester.pumpAndSettle();

    expect(sentQuestions, ['compressor checks', 'compressor checks']);
    expect(provider.messages.where((message) => message.isUser).length, 1);
    expect(_markdownContaining('Recovered.'), findsOneWidget);
    provider.dispose();
  });

  testWidgets('failed chat does not claim retrieval returned no evidence',
      (WidgetTester tester) async {
    _setDesktopViewport(tester);
    ApiService.client = MockClient(
        (request) async => http.Response(jsonEncode(_health()), 200));
    addTearDown(ApiService.client.close);
    final provider = ChatProvider();
    provider.activeSession.messages.add(ChatMessage(
      text: 'Start Ollama and retry.',
      isUser: false,
      failed: true,
      errorCode: 'language_model_unavailable',
    ));
    await tester.pumpWidget(ChangeNotifierProvider.value(
        value: provider,
        child: const MaterialApp(home: Scaffold(body: RightPanel()))));
    await tester.pumpAndSettle();

    expect(find.text('Evidence status unavailable'), findsOneWidget);
    expect(find.text('No relevant evidence found'), findsNothing);
    provider.dispose();
  });

  testWidgets('completed empty retrieval is labelled as no relevant evidence',
      (WidgetTester tester) async {
    _setDesktopViewport(tester);
    ApiService.client = MockClient(
        (request) async => http.Response(jsonEncode(_health()), 200));
    addTearDown(ApiService.client.close);
    final provider = ChatProvider();
    provider.activeSession.messages.add(ChatMessage(
      text: 'No matching record was found.',
      isUser: false,
      evidenceStatus: 'no_relevant_evidence',
    ));
    await tester.pumpWidget(ChangeNotifierProvider.value(
        value: provider,
        child: const MaterialApp(home: Scaffold(body: RightPanel()))));
    await tester.pumpAndSettle();

    expect(find.text('No relevant evidence found'), findsOneWidget);
    expect(find.text('Evidence status unavailable'), findsNothing);
    provider.dispose();
  });

  testWidgets(
      'assistant Markdown renders spaced headings and lists at narrow width',
      (WidgetTester tester) async {
    tester.view.physicalSize = const Size(360, 900);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    const answer =
        '### Conclusion\n\nThe classifier detected no failure for these inputs.\n\n### What is missing\n\n- Current timestamped trends.\n- Verified symptoms.\n\nA long technical explanation should wrap across the narrow conversation width without horizontal overflow.';

    await tester.pumpWidget(const MaterialApp(
      home: Scaffold(
        body: Center(
          child: SizedBox(
            width: 330,
            child: MessageBubble(text: answer, isUser: false),
          ),
        ),
      ),
    ));

    expect(_markdownContaining('### Conclusion'), findsOneWidget);
    expect(_markdownContaining('### What is missing'), findsOneWidget);
    expect(_markdownContaining('- Current timestamped trends'), findsOneWidget);
    expect(find.textContaining('Current timestamped trends'), findsOneWidget);
    expect(find.textContaining('Verified symptoms'), findsOneWidget);
    final markdown =
        tester.widget<MarkdownBody>(_markdownContaining('### Conclusion'));
    expect(markdown.styleSheet?.h3?.fontWeight, FontWeight.w700);
    expect(markdown.styleSheet?.h3Padding?.top, greaterThan(0));
    expect(answer, contains('\n\n- Current timestamped trends'));
    expect(
      find.byWidgetPredicate(
        (widget) => widget is SelectableText && widget.data == answer,
      ),
      findsNothing,
    );
    expect(tester.takeException(), isNull);
  });

  testWidgets(
      'chat disables empty send and rejects duplicate in-flight request',
      (WidgetTester tester) async {
    _setDesktopViewport(tester);
    final response = Completer<http.Response>();
    var chatRequests = 0;
    ApiService.client = MockClient((request) async {
      if (request.url.path.endsWith('/chat')) {
        chatRequests++;
        return response.future;
      }
      if (request.url.path.endsWith('/health')) {
        return http.Response(jsonEncode(_health()), 200);
      }
      return http.Response('{}', 404);
    });
    addTearDown(ApiService.client.close);

    await tester.pumpWidget(ChangeNotifierProvider(
      create: (_) => ChatProvider(),
      child: const MaterialApp(home: Scaffold(body: ChatArea())),
    ));
    await tester.pump();
    expect(find.textContaining('Experimental opt-in'), findsNothing);
    expect(find.textContaining('/experimental-pdm'), findsNothing);
    final sendButton = find.ancestor(
        of: find.byIcon(Icons.send_rounded),
        matching: find.byType(ElevatedButton));
    expect(tester.widget<ElevatedButton>(sendButton).onPressed, isNull);

    await tester.enterText(find.byType(TextField), 'first question');
    await tester.pump();
    await tester.tap(find.byIcon(Icons.send_rounded));
    await tester.pump();
    expect(chatRequests, 1);
    expect(find.text('Searching records and preparing a response'),
        findsOneWidget);
    expect(find.text('first question'), findsOneWidget);

    final chatContext = tester.element(find.byType(ChatArea));
    await Provider.of<ChatProvider>(chatContext, listen: false)
        .sendMessage('duplicate question');
    expect(chatRequests, 1);
    expect(find.text('duplicate question'), findsNothing);

    response.complete(http.Response(
        jsonEncode({
          'answer': '### General Guidance\n\nUse the record.',
          'sources': []
        }),
        200));
    await tester.pumpAndSettle();
    expect(_markdownContaining('### General Guidance'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('malformed chat response appears as a recoverable request error',
      (WidgetTester tester) async {
    _setDesktopViewport(tester);
    ApiService.client = MockClient((request) async {
      if (request.url.path.endsWith('/chat')) {
        return http.Response('{"answer":42,"sources":[]}', 200);
      }
      if (request.url.path.endsWith('/health')) {
        return http.Response(jsonEncode(_health()), 200);
      }
      return http.Response('{}', 404);
    });
    addTearDown(ApiService.client.close);

    await tester.pumpWidget(ChangeNotifierProvider(
      create: (_) => ChatProvider(),
      child: const MaterialApp(home: Scaffold(body: ChatArea())),
    ));
    await tester.pump();
    await tester.enterText(find.byType(TextField), 'ask a question');
    await tester.pump();
    await tester.tap(find.byIcon(Icons.send_rounded));
    await tester.pumpAndSettle();

    expect(find.text('Invalid response'), findsOneWidget);
    expect(_markdownContaining('SMARTA returned an invalid response.'),
        findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('overview omits dashboard cards and navigation still works',
      (WidgetTester tester) async {
    _setDesktopViewport(tester);
    ApiService.client = MockClient((request) async {
      if (request.url.path.endsWith('/predict/metadata')) {
        return http.Response(jsonEncode(_metadata()), 200);
      }
      if (request.url.path.endsWith('/health')) {
        return http.Response(jsonEncode(_health()), 200);
      }
      if (request.url.path.endsWith('/maintenance/history')) {
        return http.Response(jsonEncode(_maintenanceHistory()), 200);
      }
      if (request.url.path.endsWith('/chat')) {
        final payload = jsonDecode(request.body) as Map<String, dynamic>;
        if (payload['question'] == 'force service error') {
          return http.Response('{"detail":"temporarily unavailable"}', 500);
        }
        return http.Response(
            jsonEncode({
              'answer': 'The source records a cleaned filter.',
              'sources': ['[1] maintenance.xlsx, row 2'],
            }),
            200);
      }
      return http.Response('{}', 404);
    });
    addTearDown(ApiService.client.close);

    await tester.pumpWidget(const SmartaApp());
    await tester.pumpAndSettle();
    expect(find.textContaining('SMARTA is a maintenance assistant'),
        findsOneWidget);
    for (final removed in [
      'Production machine classifier',
      'Open machine intelligence',
      'System status',
      'Local language model',
      'Local corpus (SQLite)',
      'Company MySQL',
      'Live sensors',
      'Indexed source passages',
      'Maintenance records',
      'Indexed sources',
      'View all',
      'SMT13 · Vacuum pressure low',
    ]) {
      expect(find.text(removed), findsNothing, reason: removed);
    }
    expect(find.text('Ask SMARTA'), findsOneWidget);
    expect(find.text('Open the workbench'), findsOneWidget);

    await tester.tap(find.text('History'));
    await tester.pumpAndSettle();
    expect(find.text('Maintenance history'), findsOneWidget);
    expect(find.text('Cleaned filter'), findsOneWidget);
    expect(find.textContaining('Vacuum pressure low'), findsOneWidget);
    final historySearch = find.byType(TextField).first;
    await tester.enterText(historySearch, 'not-a-record');
    await tester.pumpAndSettle();
    expect(find.text('No matching records'), findsOneWidget);
    await tester.enterText(historySearch, 'SMT13');
    await tester.pumpAndSettle();
    expect(find.textContaining('Vacuum pressure low'), findsOneWidget);

    await tester.tap(find.text('Assistant'));
    await tester.pumpAndSettle();
    expect(find.text('Ask SMARTA anything'), findsOneWidget);
    expect(find.textContaining('What caused SMT13'), findsOneWidget);
    await tester.enterText(
        find.byType(TextField).last, 'What action was recorded?');
    await tester.pump();
    await tester.tap(find.byIcon(Icons.send_rounded));
    await tester.pumpAndSettle();
    expect(_markdownContaining('The source records a cleaned filter.'),
        findsOneWidget);
    expect(find.text('Sources'), findsOneWidget);
    expect(find.text('[1] maintenance.xlsx, row 2'), findsNWidgets(2));
    await tester.enterText(find.byType(TextField).last, 'force service error');
    await tester.pump();
    await tester.tap(find.byIcon(Icons.send_rounded));
    await tester.pumpAndSettle();
    expect(_markdownContaining('temporarily unavailable'), findsOneWidget);
    expect(find.text('Request failed'), findsOneWidget);

    await tester.tap(find.text('Workbench'));
    await tester.pumpAndSettle();
    expect(find.text('Predictions dashboard'), findsOneWidget);
    expect(find.text('Machine Simulator'), findsOneWidget);
    expect(find.text('Model ready'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('successful prediction opens a scoped prediction chat',
      (WidgetTester tester) async {
    _setDesktopViewport(tester);
    Map<String, dynamic>? chatPayload;
    ApiService.client = MockClient((request) async {
      if (request.url.path.endsWith('/predict/metadata')) {
        return http.Response(jsonEncode(_metadata()), 200);
      }
      if (request.url.path.endsWith('/health')) {
        return http.Response(jsonEncode(_health()), 200);
      }
      if (request.url.path.endsWith('/maintenance/history')) {
        return http.Response(jsonEncode(_maintenanceHistory()), 200);
      }
      if (request.url.path.endsWith('/predict')) {
        return http.Response(jsonEncode(_predictionResponse()), 200);
      }
      if (request.url.path.endsWith('/chat')) {
        chatPayload = jsonDecode(request.body) as Map<String, dynamic>;
        return http.Response(
            jsonEncode({
              'answer':
                  'The classifier output is uncalibrated. No diagnosis is established.',
              'sources': [],
            }),
            200);
      }
      return http.Response('{}', 404);
    });
    addTearDown(ApiService.client.close);

    await tester.pumpWidget(const SmartaApp());
    await tester.pumpAndSettle();
    await tester.tap(find.text('Workbench'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Machine Simulator'));
    await tester.pumpAndSettle();
    expect(find.byKey(const ValueKey('ask-ai-about-prediction')), findsNothing);
    await tester.ensureVisible(find.text('Run Prediction'));
    await tester.pumpAndSettle();
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.text('Run Prediction'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Run Prediction'));
    await tester.pumpAndSettle();
    expect(
        find.byKey(const ValueKey('ask-ai-about-prediction')), findsOneWidget);
    expect(
      tester
          .getTopLeft(find.byKey(const ValueKey('ask-ai-about-prediction')))
          .dy,
      greaterThan(
          tester.getTopLeft(find.text('Overall machine assessment')).dy),
      reason:
          'The action belongs immediately after the assessment in the results flow.',
    );
    await tester
        .ensureVisible(find.byKey(const ValueKey('ask-ai-about-prediction')));
    await tester.tap(find.byKey(const ValueKey('ask-ai-about-prediction')));
    await tester.pumpAndSettle();
    expect(find.textContaining('Prediction for Machine-01'), findsOneWidget);
    expect(find.textContaining('not live sensor monitoring'), findsOneWidget);
    expect(find.text('Ask about this prediction'), findsOneWidget);
    for (var index = 0; index < 4; index++) {
      expect(
        find.byKey(ValueKey('prediction-suggestion-$index')),
        findsOneWidget,
        reason: 'Prediction prompt $index must be rendered as an active chip.',
      );
    }
    expect(find.text("Explain this machine's prediction in simple terms."),
        findsOneWidget);
    expect(find.text('Which operating conditions are relevant to this result?'),
        findsOneWidget);
    expect(
        find.text(
            "Based on SMARTA's maintenance records, what action is supported? Cite the relevant source."),
        findsOneWidget);
    expect(
        find.text(
            'What information is missing before a corrective action can be recommended?'),
        findsOneWidget);
    await tester.tap(find.byKey(const ValueKey('prediction-suggestion-0')));
    await tester.pumpAndSettle();
    expect(chatPayload?['question'],
        "Explain this machine's prediction in simple terms.");
    final passedContext =
        chatPayload?['prediction_context'] as Map<String, dynamic>?;
    final expectedPrediction =
        _predictionResponse()['prediction'] as Map<String, dynamic>;
    expect(
      passedContext,
      {
        'provenance': 'production_ai4i',
        'experimental': false,
        'model_id': 'ai4i-rf-v1',
        ...expectedPrediction,
      },
      reason: 'Every API-returned prediction field must be bound to this chat.',
    );
    expect(passedContext?['machine_id'], 'Machine-01');
    expect(passedContext?['model_version'], 'ai4i-rf-v1');
    expect(passedContext?['machine_failure']['score'], 0.78);
    expect(passedContext?['input_features']['Torque [Nm]'], 42.8);
    expect(passedContext?.containsKey('threshold'), isFalse);
    expect(_markdownContaining('The classifier output is uncalibrated'),
        findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets(
      'ordinary AI Assistant stays context-free after a prediction is run',
      (WidgetTester tester) async {
    _setDesktopViewport(tester);
    Map<String, dynamic>? chatPayload;
    ApiService.client = MockClient((request) async {
      if (request.url.path.endsWith('/predict/metadata')) {
        return http.Response(jsonEncode(_metadata()), 200);
      }
      if (request.url.path.endsWith('/health')) {
        return http.Response(jsonEncode(_health()), 200);
      }
      if (request.url.path.endsWith('/maintenance/history')) {
        return http.Response(jsonEncode(_maintenanceHistory()), 200);
      }
      if (request.url.path.endsWith('/predict')) {
        return http.Response(jsonEncode(_predictionResponse()), 200);
      }
      if (request.url.path.endsWith('/chat')) {
        chatPayload = jsonDecode(request.body) as Map<String, dynamic>;
        return http.Response(
            jsonEncode(
                {'answer': 'Explained this exact result.', 'sources': []}),
            200);
      }
      return http.Response('{}', 404);
    });
    addTearDown(ApiService.client.close);

    await tester.pumpWidget(const SmartaApp());
    await tester.pumpAndSettle();
    await tester.tap(find.text('Workbench'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Machine Simulator'));
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.text('Run Prediction'));
    await tester.pumpAndSettle();
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.text('Run Prediction'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Run Prediction'));
    await tester.pumpAndSettle();

    // Direct navigation remains an ordinary, context-free conversation.
    await tester.tap(find.text('Assistant'));
    await tester.pumpAndSettle();
    expect(find.byKey(const ValueKey('open-latest-prediction-chat')),
        findsNothing);
    expect(find.text('Discuss prediction'), findsNothing);
    expect(find.text('Ask about this prediction'), findsNothing);
    await tester.enterText(
        find.byType(TextField).last, 'What maintenance records are available?');
    await tester.pump();
    await tester.tap(find.byIcon(Icons.send_rounded));
    await tester.pumpAndSettle();

    expect(chatPayload?['question'], 'What maintenance records are available?');
    expect(chatPayload?.containsKey('prediction_context'), isFalse);
    expect(find.text('Explained this exact result.'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('mobile navigation fits and opens simulated machine inputs',
      (WidgetTester tester) async {
    tester.view.physicalSize = const Size(780, 1688);
    tester.view.devicePixelRatio = 2;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    ApiService.client = MockClient((request) async {
      if (request.url.path.endsWith('/predict/metadata')) {
        return http.Response(jsonEncode(_metadata()), 200);
      }
      if (request.url.path.endsWith('/health')) {
        return http.Response(jsonEncode(_health()), 200);
      }
      if (request.url.path.endsWith('/maintenance/history')) {
        return http.Response(jsonEncode(_maintenanceHistory()), 200);
      }
      return http.Response('{}', 404);
    });
    addTearDown(ApiService.client.close);

    await tester.pumpWidget(const SmartaApp());
    await tester.pumpAndSettle();
    expect(tester.takeException(), isNull);
    expect(find.byType(NavigationBar), findsNothing);
    expect(find.text('Production machine classifier'), findsNothing);
    expect(find.text('System status'), findsNothing);
    expect(find.text('View all'), findsNothing);
    await tester.tap(find.byTooltip('Navigate SMARTA'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Workbench'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Machine Simulator'));
    await tester.pumpAndSettle();
    expect(find.text('Simulated operating point — not live sensor data'),
        findsOneWidget);
    expect(find.text('Run Prediction'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('overview and history show honest API error states',
      (WidgetTester tester) async {
    _setDesktopViewport(tester);
    ApiService.client = MockClient((request) async {
      return http.Response('{"detail":"service unavailable"}', 503);
    });
    addTearDown(ApiService.client.close);

    await tester.pumpWidget(const SmartaApp());
    await tester.pumpAndSettle();
    expect(find.textContaining('SMARTA is a maintenance assistant'),
        findsOneWidget);
    expect(find.text('AI4I model unavailable'), findsNothing);
    expect(find.text('System status unavailable'), findsNothing);
    await tester.tap(find.text('History'));
    await tester.pumpAndSettle();
    expect(find.text('Maintenance history unavailable'), findsOneWidget);
    expect(find.text('Retry'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets(
      'simulator sends entered conditions and renders API response verbatim',
      (WidgetTester tester) async {
    _setDesktopViewport(tester);
    Map<String, dynamic>? submitted;
    var predictionCalls = 0;
    ApiService.client = MockClient((request) async {
      if (request.url.path.endsWith('/predict/metadata')) {
        return http.Response(jsonEncode(_metadata()), 200);
      }
      if (request.url.path.endsWith('/predict')) {
        predictionCalls++;
        submitted = jsonDecode(request.body) as Map<String, dynamic>;
        if (predictionCalls > 1) {
          return http.Response('{"detail":"temporary inference error"}', 503);
        }
        return http.Response(jsonEncode(_predictionResponse()), 200);
      }
      return http.Response('{}', 404);
    });
    addTearDown(ApiService.client.close);

    await tester.pumpWidget(const MaterialApp(home: PredictScreen()));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Machine Simulator'));
    await tester.pumpAndSettle();
    await tester.enterText(find.byType(TextFormField).at(1), '300.5');
    await tester.ensureVisible(find.text('Run Prediction'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Run Prediction'));
    await tester.pumpAndSettle();

    expect(submitted, isNotNull);
    expect(submitted!['machine_id'], 'Machine-01');
    expect(submitted!['features']['air_temperature_k'], 300.5);
    expect(submitted!['features']['product_type'], 'M');
    expect(
        submitted!['features'].keys,
        containsAll([
          'product_type',
          'air_temperature_k',
          'process_temperature_k',
          'rotational_speed_rpm',
          'torque_nm',
          'tool_wear_min',
        ]));
    expect(find.text('Prediction request: Succeeded'), findsOneWidget);
    expect(
        find.text(
            'Machine failure classification: Failure detected by classifier'),
        findsOneWidget);
    expect(find.textContaining('Raw uncalibrated score: 0.780000'),
        findsOneWidget);
    expect(find.textContaining('Raw score: 0.490000'), findsOneWidget);
    expect(find.textContaining('Raw score: 0.520000'), findsOneWidget);
    expect(find.text('Not classified'), findsOneWidget);
    expect(find.text('Classified'), findsOneWidget);
    await tester.ensureVisible(find.textContaining(
        'The available documents did not support a corrective action'));
    await tester.pumpAndSettle();
    expect(
        find.textContaining(
            'The available documents did not support a corrective action'),
        findsOneWidget);
    expect(find.textContaining('No sufficiently supported corrective action'),
        findsOneWidget);
    await tester.enterText(find.byType(TextFormField).at(1), '299.0');
    await tester.ensureVisible(find.text('Run Prediction'));
    await tester.pumpAndSettle();
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.text('Run Prediction'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Run Prediction'));
    await tester.pumpAndSettle();
    expect(predictionCalls, 2);
    expect(find.text('Prediction request: Failed'), findsOneWidget);
    expect(find.text('temporary inference error'), findsOneWidget);
    expect(find.text('Model prediction'), findsNothing);
    await tester.ensureVisible(find.text('Reset'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Reset'));
    await tester.pumpAndSettle();
    expect(
        tester
            .widget<TextFormField>(find.byType(TextFormField).at(1))
            .controller
            ?.text,
        '298.1');
    expect(find.text('Model prediction'), findsNothing);
  });

  testWidgets(
      'simulator accepts out-of-training-range input and warns before inference',
      (WidgetTester tester) async {
    _setDesktopViewport(tester);
    var predictionCalls = 0;
    ApiService.client = MockClient((request) async {
      if (request.url.path.endsWith('/predict/metadata')) {
        return http.Response(jsonEncode(_metadata()), 200);
      }
      if (request.url.path.endsWith('/predict')) {
        predictionCalls++;
        final response = _predictionResponse();
        (response['prediction'] as Map<String, dynamic>)['input_features'] = {
          'Air temperature [K]': 310.0,
        };
        return http.Response(jsonEncode(response), 200);
      }
      return http.Response('{}', 404);
    });
    addTearDown(ApiService.client.close);

    await tester.pumpWidget(const MaterialApp(home: PredictScreen()));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Machine Simulator'));
    await tester.pumpAndSettle();
    await tester.enterText(find.byType(TextFormField).at(1), '310');
    await tester.ensureVisible(find.text('Run Prediction'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Run Prediction'));
    await tester.pumpAndSettle();

    expect(predictionCalls, 1);
    expect(find.textContaining('Outside training range; inference is allowed'),
        findsOneWidget);
    expect(find.text('Outside AI4I training range'), findsOneWidget);
    expect(find.text('Prediction request: Succeeded'), findsOneWidget);
  });

  testWidgets('process temperature below 300 K is submitted without clamping',
      (WidgetTester tester) async {
    _setDesktopViewport(tester);
    Map<String, dynamic>? sent;
    ApiService.client = MockClient((request) async {
      if (request.url.path.endsWith('/predict/metadata')) {
        return http.Response(jsonEncode(_metadata()), 200);
      }
      if (request.url.path.endsWith('/predict')) {
        sent = jsonDecode(request.body) as Map<String, dynamic>;
        final response = _predictionResponse();
        (response['prediction'] as Map<String, dynamic>)['input_features'] = {
          'Process temperature [K]': 299.0,
        };
        return http.Response(jsonEncode(response), 200);
      }
      return http.Response('{}', 404);
    });
    addTearDown(ApiService.client.close);

    await tester.pumpWidget(const MaterialApp(home: PredictScreen()));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Machine Simulator'));
    await tester.pumpAndSettle();
    await tester.enterText(find.byType(TextFormField).at(2), '299');
    await tester.ensureVisible(find.text('Run Prediction'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Run Prediction'));
    await tester.pumpAndSettle();

    expect(sent, isNotNull);
    final features = sent!['features'] as Map<String, dynamic>;
    expect(features['process_temperature_k'], 299);
    expect(find.text('Prediction request: Succeeded'), findsOneWidget);
    expect(find.textContaining('Outside training range; inference is allowed'),
        findsOneWidget);
    expect(find.text('Outside AI4I training range'), findsOneWidget);
  });

  testWidgets(
      'simulator distinguishes missing, malformed, non-finite and invalid values',
      (WidgetTester tester) async {
    _setDesktopViewport(tester);
    var predictionCalls = 0;
    ApiService.client = MockClient((request) async {
      if (request.url.path.endsWith('/predict/metadata')) {
        return http.Response(jsonEncode(_metadata()), 200);
      }
      if (request.url.path.endsWith('/predict')) predictionCalls++;
      return http.Response('{}', 404);
    });
    addTearDown(ApiService.client.close);

    await tester.pumpWidget(const MaterialApp(home: PredictScreen()));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Machine Simulator'));
    await tester.pumpAndSettle();
    final airField = find.byType(TextFormField).at(1);

    for (final check in <(String, String)>[
      ('', 'This value is required'),
      ('not-a-number', 'Enter a valid number'),
      ('NaN', 'Enter a finite number'),
      ('0', 'Temperature must be above absolute zero (0 K)'),
    ]) {
      await tester.enterText(airField, check.$1);
      await tester.ensureVisible(find.text('Run Prediction'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Run Prediction'));
      await tester.pumpAndSettle();
      expect(find.text(check.$2), findsOneWidget);
    }
    expect(predictionCalls, 0);
  });

  testWidgets(
      'loading the synthetic example populates inputs without inference',
      (WidgetTester tester) async {
    _setDesktopViewport(tester);
    var predictionCalls = 0;
    ApiService.client = MockClient((request) async {
      if (request.url.path.endsWith('/predict/metadata')) {
        return http.Response(jsonEncode(_metadata()), 200);
      }
      if (request.url.path.endsWith('/predict')) predictionCalls++;
      return http.Response('{}', 404);
    });
    addTearDown(ApiService.client.close);

    await tester.pumpWidget(const MaterialApp(home: PredictScreen()));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Machine Simulator'));
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.text('Load synthetic example'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Load synthetic example'));
    await tester.pumpAndSettle();

    expect(
        tester
            .widget<DropdownButtonFormField<String>>(
                find.byType(DropdownButtonFormField<String>))
            .initialValue,
        'H');
    expect(
        tester
            .widget<TextFormField>(find.byType(TextFormField).at(1))
            .controller
            ?.text,
        '304.5');
    expect(
        tester
            .widget<TextFormField>(find.byType(TextFormField).at(2))
            .controller
            ?.text,
        '313.8');
    expect(
        tester
            .widget<TextFormField>(find.byType(TextFormField).at(3))
            .controller
            ?.text,
        '1168');
    expect(
        tester
            .widget<TextFormField>(find.byType(TextFormField).at(4))
            .controller
            ?.text,
        '76.6');
    expect(
        tester
            .widget<TextFormField>(find.byType(TextFormField).at(5))
            .controller
            ?.text,
        '253');
    expect(find.text('Prediction request: Not run'), findsOneWidget);
    expect(predictionCalls, 0);
  });

  testWidgets('RAG evidence references map to the returned source passage',
      (WidgetTester tester) async {
    _setDesktopViewport(tester);
    final response = _predictionResponse()
      ..['evidence'] = [
        {
          'source': 'manual.pdf',
          'locator': 'page 2',
          'text': 'Inspect the ventilation path for blockage.',
        },
      ]
      ..['recommendation'] = 'The manual recommends checking ventilation. [1]'
      ..['evidence_status'] = 'general_only';
    ApiService.client = MockClient((request) async {
      if (request.url.path.endsWith('/predict/metadata')) {
        return http.Response(jsonEncode(_metadata()), 200);
      }
      if (request.url.path.endsWith('/predict')) {
        return http.Response(jsonEncode(response), 200);
      }
      return http.Response('{}', 404);
    });
    addTearDown(ApiService.client.close);

    await tester.pumpWidget(const MaterialApp(home: PredictScreen()));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Machine Simulator'));
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.text('Run Prediction'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Run Prediction'));
    await tester.pumpAndSettle();

    await tester.ensureVisible(
        find.text('The manual recommends checking ventilation. [1]'));
    await tester.pumpAndSettle();

    expect(find.text('The manual recommends checking ventilation. [1]'),
        findsOneWidget);
    expect(find.text('manual.pdf · page 2'), findsOneWidget);
    expect(find.text('Inspect the ventilation path for blockage.'),
        findsOneWidget);
  });

  testWidgets('model and prediction API errors are visible to the user',
      (WidgetTester tester) async {
    _setDesktopViewport(tester);
    ApiService.client = MockClient((request) async => http.Response(
        '{"detail":"The predictive model is not trained."}', 503));
    addTearDown(ApiService.client.close);

    await tester.pumpWidget(const MaterialApp(home: PredictScreen()));
    await tester.pumpAndSettle();
    expect(find.text('Model unavailable'), findsOneWidget);
    await tester.tap(find.text('Machine Simulator'));
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.text('Run Prediction'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Run Prediction'));
    await tester.pumpAndSettle();
    expect(find.text('Prediction request: Failed'), findsOneWidget);
    expect(find.text('The predictive model is not trained.'), findsOneWidget);
  });

  testWidgets(
      'malformed successful HTTP response is shown as inference failure',
      (WidgetTester tester) async {
    _setDesktopViewport(tester);
    ApiService.client = MockClient((request) async {
      if (request.url.path.endsWith('/predict/metadata')) {
        return http.Response(jsonEncode(_metadata()), 200);
      }
      if (request.url.path.endsWith('/predict')) {
        return http.Response('{"prediction":{}}', 200);
      }
      return http.Response('{}', 404);
    });
    addTearDown(ApiService.client.close);

    await tester.pumpWidget(const MaterialApp(home: PredictScreen()));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Machine Simulator'));
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.text('Run Prediction'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Run Prediction'));
    await tester.pumpAndSettle();

    expect(find.text('Prediction request: Failed'), findsOneWidget);
    expect(
        find.textContaining('missing machine-failure output'), findsOneWidget);
    expect(find.text('Model prediction'), findsNothing);
    expect(tester.takeException(), isNull);
  });
}
