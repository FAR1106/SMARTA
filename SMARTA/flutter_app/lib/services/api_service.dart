import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

class ChatReply {
  final String answer;
  final List<String> sources;
  final bool failed;
  final String? errorCode;
  final String evidenceStatus;

  const ChatReply({
    required this.answer,
    this.sources = const [],
    this.failed = false,
    this.errorCode,
    this.evidenceStatus = 'unknown',
  });
}

class ApiService {
  // Desktop / Web (running the FastAPI server on the same machine)
  static String baseUrl = const String.fromEnvironment(
    'SMARTA_API_BASE_URL',
    defaultValue: 'http://localhost:8000',
  );

  // Android Emulator
  // static String baseUrl = "http://10.0.2.2:8000";

  // Physical Device (replace with your machine's LAN IP)
  // static String baseUrl = "http://192.168.1.15:8000";
  static http.Client client = http.Client();

  static Future<ChatReply> sendMessage(
    String question, {
    Map<String, dynamic>? predictionContext,
  }) async {
    final stopwatch = Stopwatch()..start();
    void logTiming(String outcome) {
      if (kDebugMode) {
        debugPrint(
          'smarta_chat_client_timing elapsed_ms=${stopwatch.elapsedMilliseconds} outcome=$outcome',
        );
      }
    }

    try {
      final response = await client
          .post(
            Uri.parse("$baseUrl/chat"),
            headers: {
              "Content-Type": "application/json",
            },
            body: jsonEncode({
              "question": question,
              if (predictionContext != null)
                "prediction_context": predictionContext,
            }),
          )
          .timeout(const Duration(seconds: 60));

      if (response.statusCode == 200) {
        final decoded = jsonDecode(response.body);
        if (decoded is! Map<String, dynamic> ||
            decoded['answer'] is! String ||
            (decoded['sources'] != null && decoded['sources'] is! List)) {
          logTiming('invalid_response');
          return const ChatReply(
            answer: 'SMARTA returned an invalid response. Please try again.',
            failed: true,
            errorCode: 'invalid_response',
          );
        }

        logTiming('success');
        return ChatReply(
          answer: decoded["answer"] as String,
          sources: (decoded["sources"] as List<dynamic>? ?? const [])
              .whereType<String>()
              .toList(),
          evidenceStatus: decoded['evidence_status'] is String
              ? decoded['evidence_status'] as String
              : 'unknown',
        );
      }

      final body = jsonDecode(response.body);
      final detail = body is Map<String, dynamic> ? body["detail"] : null;
      final errorCode =
          detail is Map<String, dynamic> ? detail['code']?.toString() : null;
      final message =
          detail is Map<String, dynamic> ? detail['message'] : detail;
      logTiming('http_error_${response.statusCode}');
      return ChatReply(
        answer: message is String
            ? message
            : "SMARTA returned server error ${response.statusCode}.",
        failed: true,
        errorCode: errorCode ?? 'api_request_failed',
      );
    } on TimeoutException {
      logTiming('api_request_timeout');
      return const ChatReply(
        answer: 'The SMARTA API request timed out. Please retry.',
        failed: true,
        errorCode: 'api_request_timeout',
      );
    } on FormatException {
      logTiming('invalid_response');
      return const ChatReply(
        answer: 'SMARTA returned an invalid response. Please try again.',
        failed: true,
        errorCode: 'invalid_response',
      );
    } catch (e) {
      logTiming('transport_or_decode_error');
      return const ChatReply(
        answer:
            "Unable to reach the SMARTA API. Check its connection and try again.",
        failed: true,
        errorCode: 'api_request_failed',
      );
    }
  }

  static Future<Map<String, dynamic>> getSystemHealth() async {
    final response = await client
        .get(Uri.parse('$baseUrl/health'))
        .timeout(const Duration(seconds: 5));
    final body = jsonDecode(response.body);
    if (response.statusCode != 200 || body is! Map<String, dynamic>) {
      throw Exception('System status unavailable (${response.statusCode}).');
    }
    return body;
  }

  static Future<Map<String, dynamic>> getMaintenanceHistory({
    int limit = 50,
    int offset = 0,
  }) async {
    final response = await client
        .get(Uri.parse(
            '$baseUrl/maintenance/history?limit=$limit&offset=$offset'))
        .timeout(const Duration(seconds: 10));
    final body = jsonDecode(response.body);
    if (response.statusCode != 200 || body is! Map<String, dynamic>) {
      final detail = body is Map<String, dynamic> ? body['detail'] : null;
      throw Exception(detail ??
          'Maintenance history unavailable (${response.statusCode}).');
    }
    return body;
  }

  static Future<Map<String, dynamic>> predict({
    required String machineId,
    required String productType,
    required double airTemperature,
    required double processTemperature,
    required double rotationalSpeed,
    required double torque,
    required double toolWear,
  }) async {
    final response = await client
        .post(
          Uri.parse("$baseUrl/predict"),
          headers: {"Content-Type": "application/json"},
          body: jsonEncode({
            "machine_id": machineId,
            "observed_at": DateTime.now().toUtc().toIso8601String(),
            "features": {
              "product_type": productType,
              "air_temperature_k": airTemperature,
              "process_temperature_k": processTemperature,
              "rotational_speed_rpm": rotationalSpeed,
              "torque_nm": torque,
              "tool_wear_min": toolWear,
            },
          }),
        )
        .timeout(const Duration(seconds: 60));
    final body = jsonDecode(response.body);
    if (response.statusCode != 200 || body is! Map<String, dynamic>) {
      final detail = body is Map<String, dynamic> ? body["detail"] : null;
      throw Exception(detail ?? "Prediction failed (${response.statusCode}).");
    }
    _validatePredictionResponse(body);
    return body;
  }

  static void _validatePredictionResponse(Map<String, dynamic> body) {
    final prediction = body['prediction'];
    if (prediction is! Map<String, dynamic>) {
      throw Exception('Prediction response is missing the prediction object.');
    }
    if (prediction['machine_failure'] is! Map<String, dynamic>) {
      throw Exception('Prediction response is missing machine-failure output.');
    }
    final machine = prediction['machine_failure'] as Map<String, dynamic>;
    if (machine['classification'] is! num || machine['score'] is! num) {
      throw Exception(
          'Prediction response has malformed machine-failure output.');
    }
    if (prediction['failure_modes'] is! Map<String, dynamic>) {
      throw Exception('Prediction response is missing failure-mode output.');
    }
    final modes = prediction['failure_modes'] as Map<String, dynamic>;
    for (final entry in modes.entries) {
      if (entry.value is! Map<String, dynamic>) {
        throw Exception(
            'Prediction response has malformed ${entry.key} output.');
      }
      final output = entry.value as Map<String, dynamic>;
      if (output['classification'] is! num || output['score'] is! num) {
        throw Exception(
            'Prediction response has malformed ${entry.key} output.');
      }
    }
    if (prediction['model_version'] is! String ||
        prediction['prediction_timestamp'] is! String) {
      throw Exception(
          'Prediction response is missing model version or inference time.');
    }
    if (body['evidence'] is! List ||
        body['machine_specific_evidence'] is! List ||
        body['recommendation'] is! String ||
        body['evidence_status'] is! String) {
      throw Exception('Prediction response has malformed diagnostic evidence.');
    }
  }

  static Future<Map<String, dynamic>> getPredictiveModelMetadata() async {
    final response = await client
        .get(Uri.parse('$baseUrl/predict/metadata'))
        .timeout(const Duration(seconds: 10));
    final body = jsonDecode(response.body);
    if (response.statusCode != 200 || body is! Map<String, dynamic>) {
      final detail = body is Map<String, dynamic> ? body['detail'] : null;
      throw Exception(
          detail ?? 'Model status unavailable (${response.statusCode}).');
    }
    return body;
  }

  /// Pings the backend's /health endpoint (defined in api.py).
  static Future<bool> pingServer() async {
    try {
      final response = await client
          .get(Uri.parse("$baseUrl/health"))
          .timeout(const Duration(seconds: 5));

      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  static Future<String> getServerStatus() async {
    final online = await pingServer();

    return online ? "Connected" : "Disconnected";
  }

  static void changeServer(String url) {
    baseUrl = url;
  }
}
