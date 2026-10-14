import 'package:flutter/material.dart';

import '../services/api_service.dart';
import '../models/prediction_chat_context.dart';
import '../theme.dart';

class PredictScreen extends StatefulWidget {
  final bool embedded;
  final ValueChanged<PredictionChatContext>? onAskAboutPrediction;

  const PredictScreen({
    super.key,
    this.embedded = false,
    this.onAskAboutPrediction,
  });

  @override
  State<PredictScreen> createState() => _PredictScreenState();
}

class _PredictScreenState extends State<PredictScreen> {
  static const _defaults = <String, String>{
    'air': '298.1',
    'process': '308.6',
    'speed': '1551',
    'torque': '42.8',
    'wear': '20',
  };
  static const _fallbackRanges = <String, List<double>>{
    'Air temperature [K]': [295.3, 304.5],
    'Process temperature [K]': [305.7, 313.8],
    'Rotational speed [rpm]': [1168, 2886],
    'Torque [Nm]': [3.8, 76.6],
    'Tool wear [min]': [0, 253],
  };
  static const _failureModeNames = <String, String>{
    'TWF': 'Tool wear failure',
    'HDF': 'Heat dissipation failure',
    'PWF': 'Power failure',
    'OSF': 'Overstrain failure',
    'RNF': 'Random failure',
  };

  final _formKey = GlobalKey<FormState>();
  final _machine = TextEditingController(text: 'Machine-01');
  final _air = TextEditingController(text: _defaults['air']);
  final _process = TextEditingController(text: _defaults['process']);
  final _speed = TextEditingController(text: _defaults['speed']);
  final _torque = TextEditingController(text: _defaults['torque']);
  final _wear = TextEditingController(text: _defaults['wear']);
  late final Future<Map<String, dynamic>> _metadataFuture;
  Map<String, dynamic>? _metadata;
  String _type = 'M';
  bool _loading = false;
  String? _error;
  Map<String, dynamic>? _result;
  String _requestStatus = 'Not run';

  @override
  void initState() {
    super.initState();
    _metadataFuture = ApiService.getPredictiveModelMetadata().then((metadata) {
      if (mounted) setState(() => _metadata = metadata);
      return metadata;
    });
  }

  @override
  void dispose() {
    for (final controller in [
      _machine,
      _air,
      _process,
      _speed,
      _torque,
      _wear,
    ]) {
      controller.dispose();
    }
    super.dispose();
  }

  double? _number(String? value) => double.tryParse(value?.trim() ?? '');

  Map<String, List<double>> get _ranges {
    final raw = _metadata?['feature_ranges'];
    if (raw is! Map<String, dynamic>) return _fallbackRanges;
    return raw.map((key, value) => MapEntry(
          key,
          (value as List<dynamic>)
              .map((item) => (item as num).toDouble())
              .toList(),
        ));
  }

  String? _validateNumber(String? input, String featureName) {
    if (input == null || input.trim().isEmpty) return 'This value is required';
    final value = _number(input);
    if (value == null) return 'Enter a valid number';
    if (!value.isFinite) return 'Enter a finite number';
    if ((featureName == 'Air temperature [K]' ||
            featureName == 'Process temperature [K]') &&
        value <= 0) {
      return 'Temperature must be above absolute zero (0 K)';
    }
    if (featureName == 'Tool wear [min]' && value < 0) {
      return 'Tool wear cannot be negative';
    }
    return null;
  }

  bool _outsideTrainingRange(String featureName, String input) {
    final value = _number(input);
    final range = _ranges[featureName];
    return value != null &&
        value.isFinite &&
        range != null &&
        (value < range[0] || value > range[1]);
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) {
      setState(() {
        _result = null;
        _error = null;
        _requestStatus = 'Validation error';
      });
      return;
    }
    setState(() {
      _loading = true;
      _requestStatus = 'Running';
      _error = null;
      _result = null;
    });
    try {
      final response = await ApiService.predict(
        machineId: _machine.text.trim(),
        productType: _type,
        airTemperature: _number(_air.text)!,
        processTemperature: _number(_process.text)!,
        rotationalSpeed: _number(_speed.text)!,
        torque: _number(_torque.text)!,
        toolWear: _number(_wear.text)!,
      );
      if (mounted) {
        setState(() {
          _result = response;
          _requestStatus = 'Succeeded';
        });
      }
    } catch (error) {
      if (mounted) {
        setState(
          () {
            _error = error.toString().replaceFirst('Exception: ', '');
            _requestStatus = 'Failed';
          },
        );
      }
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  void _reset() {
    setState(() {
      _machine.text = 'Machine-01';
      _air.text = _defaults['air']!;
      _process.text = _defaults['process']!;
      _speed.text = _defaults['speed']!;
      _torque.text = _defaults['torque']!;
      _wear.text = _defaults['wear']!;
      _type = 'M';
      _error = null;
      _result = null;
      _requestStatus = 'Not run';
    });
    _formKey.currentState?.reset();
  }

  void _loadExample() {
    _formKey.currentState?.reset();
    setState(() {
      _machine.text = 'Machine-01';
      _air.text = '304.5';
      _process.text = '313.8';
      _speed.text = '1168';
      _torque.text = '76.6';
      _wear.text = '253';
      _type = 'H';
      _error = null;
      _result = null;
      _requestStatus = 'Not run';
    });
  }

  Widget _numberField(
    String label,
    String featureName,
    String unit,
    TextEditingController controller,
  ) {
    final range = _ranges[featureName] ?? _fallbackRanges[featureName]!;
    final outside = _outsideTrainingRange(featureName, controller.text);
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          TextFormField(
            controller: controller,
            keyboardType: const TextInputType.numberWithOptions(decimal: true),
            decoration: InputDecoration(
              labelText: label,
              suffixText: unit,
            ),
            validator: (value) => _validateNumber(value, featureName),
            onChanged: (_) => setState(() {}),
          ),
          const SizedBox(height: 6),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'AI4I training data range: ${range[0]}–${range[1]} $unit',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
                if (outside)
                  Text(
                    'Outside training range; inference is allowed, but this input is unfamiliar to the model.',
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: Theme.of(context).colorScheme.tertiary,
                        ),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) => DefaultTabController(
        length: 2,
        child: Builder(
          builder: (context) {
            const tabBar = TabBar(
              dividerColor: Colors.transparent,
              indicatorSize: TabBarIndicatorSize.tab,
              indicator: BoxDecoration(
                color: AppTheme.accent,
                borderRadius: BorderRadius.all(Radius.circular(24)),
              ),
              labelColor: Colors.white,
              unselectedLabelColor: AppTheme.textPrimary,
              labelStyle: TextStyle(fontWeight: FontWeight.w700),
              tabs: [
                Tab(icon: Icon(Icons.dashboard_outlined), text: 'Dashboard'),
                Tab(icon: Icon(Icons.tune), text: 'Machine Simulator'),
              ],
            );
            final tabView = TabBarView(
              children: [
                _buildDashboard(),
                _buildSimulator(),
              ],
            );
            return Scaffold(
              appBar: widget.embedded
                  ? null
                  : AppBar(
                      title: const Text('Predictive maintenance'),
                      bottom: tabBar,
                    ),
              body: widget.embedded
                  ? Column(
                      children: [
                        const Material(
                          color: AppTheme.card2,
                          child: Padding(
                            padding: EdgeInsets.symmetric(
                                horizontal: 8, vertical: 6),
                            child: tabBar,
                          ),
                        ),
                        Expanded(child: tabView),
                      ],
                    )
                  : tabView,
            );
          },
        ),
      );

  Widget _page(List<Widget> children) => Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 1360),
          child: ListView(
            padding: const EdgeInsets.fromLTRB(30, 24, 30, 34),
            children: children,
          ),
        ),
      );

  Widget _buildDashboard() => LayoutBuilder(builder: (context, constraints) {
        final desktop = constraints.maxWidth >= 1080;
        final overview =
            Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          _modelReadinessCard(),
          const SizedBox(height: 12),
          _requestStatusCard(),
          if (_error != null) ...[const SizedBox(height: 8), _errorCard()],
          const SizedBox(height: 12),
          _selectedMachineCard(),
        ]);
        final assessment = _result == null
            ? const Card(
                child: ListTile(
                    leading: Icon(Icons.insights_outlined),
                    title: Text('No prediction run yet'),
                    subtitle: Text(
                        'Open Machine Simulator to submit an operating point to the API.')))
            : Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                _buildOverallAssessment(),
                const SizedBox(height: 10),
                _askAboutPredictionAction(),
                const SizedBox(height: 10),
                _resultTrainingRangeWarning(),
                const SizedBox(height: 12),
                _buildFailureModeBreakdown(),
              ]);
        return _page([
          _pageHeading(
            leading: 'Predictions',
            accent: 'dashboard',
            subtitle:
                'Model classifications are separate from historical evidence and diagnostic guidance.',
          ),
          const SizedBox(height: 18),
          if (desktop)
            Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Expanded(flex: 5, child: overview),
              const SizedBox(width: 18),
              Expanded(flex: 6, child: assessment),
            ])
          else ...[overview, const SizedBox(height: 14), assessment],
          if (_result != null) ...[
            const SizedBox(height: 14),
            _buildRagExplanation()
          ],
        ]);
      });

  Widget _modelReadinessCard() => FutureBuilder<Map<String, dynamic>>(
        future: _metadataFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Card(
                child: ListTile(
                    leading: SizedBox.square(
                        dimension: 20,
                        child: CircularProgressIndicator(strokeWidth: 2)),
                    title: Text('Checking model readiness…')));
          }
          if (snapshot.hasError) {
            return Card(
                child: ListTile(
                    leading: const Icon(Icons.error_outline),
                    title: const Text('Model unavailable'),
                    subtitle: Text(snapshot.error
                        .toString()
                        .replaceFirst('Exception: ', ''))));
          }
          final metadata = snapshot.data!;
          final supported =
              (metadata['supported_failure_modes'] as List<dynamic>)
                  .map((mode) => mode.toString())
                  .toList();
          final omitted =
              metadata['unsupported_failure_modes'] as Map<String, dynamic>;
          return Card(
              child: Padding(
                  padding: const EdgeInsets.all(18),
                  child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(children: [
                          const Icon(Icons.check_circle,
                              color: AppTheme.success),
                          const SizedBox(width: 10),
                          const Expanded(
                              child: Text('Model ready',
                                  style:
                                      TextStyle(fontWeight: FontWeight.bold))),
                          Flexible(
                              child: Text(metadata['model_version'].toString(),
                                  maxLines: 1,
                                  overflow: TextOverflow.ellipsis,
                                  style:
                                      const TextStyle(fontFamily: 'monospace')))
                        ]),
                        const SizedBox(height: 12),
                        Text('Supported failure modes',
                            style: Theme.of(context).textTheme.labelLarge),
                        const SizedBox(height: 5),
                        Text(supported
                            .map((mode) =>
                                '$mode — ${_failureModeNames[mode] ?? mode}')
                            .join(' · ')),
                        if (omitted.isNotEmpty) ...[
                          const SizedBox(height: 8),
                          Text(
                              'Unavailable: ${omitted.entries.map((entry) => '${entry.key} — ${entry.value}').join('; ')}',
                              style: Theme.of(context).textTheme.bodySmall)
                        ],
                        const SizedBox(height: 10),
                        Text(
                            'Raw scores are uncalibrated model scores, not probabilities.',
                            style: Theme.of(context).textTheme.bodySmall),
                      ])));
        },
      );

  Widget _selectedMachineCard() => Card(
      child: Padding(
          padding: const EdgeInsets.all(16),
          child:
              Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Row(children: [
              const Icon(Icons.precision_manufacturing_outlined,
                  color: AppTheme.accent),
              const SizedBox(width: 8),
              Expanded(
                  child: Text(
                      'Selected machine: ${_machine.text.isEmpty ? 'Not set' : _machine.text}',
                      style: Theme.of(context).textTheme.titleMedium)),
              TextButton(
                  onPressed: () =>
                      DefaultTabController.of(context).animateTo(1),
                  child: const Text('Edit'))
            ]),
            const SizedBox(height: 8),
            Text(
                'Supplied operating point: Type $_type · Air ${_air.text} K · Process ${_process.text} K · ${_speed.text} rpm · ${_torque.text} Nm · Tool wear ${_wear.text} min',
                style: Theme.of(context).textTheme.bodySmall),
          ])));

  Widget _pageHeading(
          {required String leading,
          required String accent,
          required String subtitle}) =>
      Padding(
        padding: const EdgeInsets.symmetric(horizontal: 2),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text.rich(
              TextSpan(children: [
                TextSpan(text: '$leading '),
                TextSpan(
                    text: accent,
                    style: const TextStyle(
                        fontFamily: 'Georgia',
                        fontStyle: FontStyle.italic,
                        color: AppTheme.botanical,
                        fontWeight: FontWeight.w700))
              ]),
              style: Theme.of(context).textTheme.headlineLarge),
          const SizedBox(height: 6),
          Text(subtitle, style: Theme.of(context).textTheme.bodyMedium),
        ]),
      );

  Widget _buildSimulator() => LayoutBuilder(
        builder: (context, constraints) {
          final desktop = constraints.maxWidth >= 1050;
          final header = _pageHeading(
            leading: 'Machine',
            accent: 'intelligence',
            subtitle:
                'AI4I 2020 synthetic-data classification · simulated inputs only',
          );
          final resultColumn = Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _buildOverallAssessment(),
              const SizedBox(height: 10),
              _askAboutPredictionAction(),
              if (_result != null) ...[
                const SizedBox(height: 10),
                _resultTrainingRangeWarning(),
              ],
              const SizedBox(height: 14),
              _buildFailureModeBreakdown(),
              if (_result != null) ...[
                const SizedBox(height: 12),
                _buildRagExplanation(),
              ],
            ],
          );

          if (desktop) {
            return Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 1360),
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(28, 20, 28, 20),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Row(children: [
                        Expanded(child: header),
                        if (_loading)
                          const Padding(
                              padding: EdgeInsets.only(left: 12),
                              child: SizedBox.square(
                                  dimension: 22,
                                  child: CircularProgressIndicator(
                                      strokeWidth: 2)))
                      ]),
                      const SizedBox(height: 12),
                      _requestStatusCard(),
                      if (_error != null) ...[
                        const SizedBox(height: 8),
                        _errorCard()
                      ],
                      const SizedBox(height: 12),
                      Expanded(
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Expanded(
                              flex: 5,
                              child: SingleChildScrollView(
                                padding: const EdgeInsets.only(bottom: 16),
                                child: _buildInputCard(),
                              ),
                            ),
                            const SizedBox(width: 18),
                            Expanded(
                              flex: 6,
                              child: SingleChildScrollView(
                                padding: const EdgeInsets.only(bottom: 16),
                                child: resultColumn,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            );
          }

          return Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 860),
              child: ListView(
                padding: const EdgeInsets.fromLTRB(20, 18, 20, 28),
                children: [
                  header,
                  const SizedBox(height: 12),
                  _requestStatusCard(),
                  if (_error != null) ...[
                    const SizedBox(height: 8),
                    _errorCard(),
                  ],
                  const SizedBox(height: 14),
                  _buildInputCard(),
                  const SizedBox(height: 14),
                  resultColumn,
                ],
              ),
            ),
          );
        },
      );

  Widget _buildInputCard() => Card(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Operating conditions',
                    style:
                        TextStyle(fontSize: 19, fontWeight: FontWeight.w600)),
                const SizedBox(height: 4),
                Text('Enter one feature vector for classification.',
                    style: Theme.of(context).textTheme.bodySmall),
                const SizedBox(height: 12),
                const Text('Simulated operating point — not live sensor data',
                    style: TextStyle(fontWeight: FontWeight.w600)),
                const SizedBox(height: 4),
                Text(
                  'This model classifies inputs; it does not forecast a failure date.',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
                const SizedBox(height: 12),
                _inputSectionTitle('Machine identity and product type'),
                const SizedBox(height: 10),
                _fieldPair(
                  TextFormField(
                    controller: _machine,
                    decoration:
                        const InputDecoration(labelText: 'Machine identifier'),
                    validator: (value) => (value?.trim().isEmpty ?? true)
                        ? 'Enter a machine identifier'
                        : null,
                    onChanged: (_) => setState(() {}),
                  ),
                  DropdownButtonFormField<String>(
                    key: ValueKey(_type),
                    initialValue: _type,
                    decoration:
                        const InputDecoration(labelText: 'Product type'),
                    items: const [
                      DropdownMenuItem(value: 'L', child: Text('L')),
                      DropdownMenuItem(value: 'M', child: Text('M')),
                      DropdownMenuItem(value: 'H', child: Text('H')),
                    ],
                    onChanged: (value) => setState(() => _type = value ?? 'M'),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.only(top: 6),
                  child: Text(
                    'L, M, and H are AI4I dataset categories, not a universal list of industrial machine types.',
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                ),
                const SizedBox(height: 12),
                _inputSectionTitle('Thermal conditions'),
                const SizedBox(height: 10),
                _fieldPair(
                  _numberField(
                      'Air temperature', 'Air temperature [K]', 'K', _air),
                  _numberField('Process temperature', 'Process temperature [K]',
                      'K', _process),
                ),
                const SizedBox(height: 4),
                _inputSectionTitle('Mechanical conditions'),
                const SizedBox(height: 10),
                _fieldPair(
                  _numberField('Rotational speed', 'Rotational speed [rpm]',
                      'rpm', _speed),
                  _numberField('Torque', 'Torque [Nm]', 'Nm', _torque),
                ),
                const SizedBox(height: 4),
                _inputSectionTitle('Tool wear'),
                const SizedBox(height: 10),
                _numberField('Tool wear', 'Tool wear [min]', 'min', _wear),
                const SizedBox(height: 2),
                Wrap(
                  spacing: 10,
                  runSpacing: 10,
                  children: [
                    FilledButton.icon(
                      onPressed: _loading ? null : _submit,
                      icon: _loading
                          ? const SizedBox.square(
                              dimension: 16,
                              child: CircularProgressIndicator(strokeWidth: 2))
                          : const Icon(Icons.play_arrow),
                      label: Text(_loading ? 'Running…' : 'Run Prediction'),
                    ),
                    OutlinedButton.icon(
                      onPressed: _loading ? null : _reset,
                      icon: const Icon(Icons.restart_alt),
                      label: const Text('Reset'),
                    ),
                    TextButton.icon(
                      onPressed: _loading ? null : _loadExample,
                      icon: const Icon(Icons.science_outlined),
                      label: const Text('Load synthetic example'),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      );

  Widget _inputSectionTitle(String title) => Text(
        title,
        style: TextStyle(
          color: Theme.of(context).colorScheme.primary,
          fontSize: 13,
          fontWeight: FontWeight.w700,
        ),
      );

  Widget _fieldPair(Widget first, Widget second) => LayoutBuilder(
        builder: (context, constraints) => constraints.maxWidth < 430
            ? Column(children: [first, const SizedBox(height: 2), second])
            : Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(child: first),
                  const SizedBox(width: 12),
                  Expanded(child: second),
                ],
              ),
      );

  Widget _errorCard() => Card(
        color: Theme.of(context).colorScheme.errorContainer,
        child: ListTile(
          leading: const Icon(Icons.warning_amber_outlined),
          title: const Text('Prediction request failed'),
          subtitle: Text(_error!),
        ),
      );

  Widget _requestStatusCard() {
    final prediction = _result?['prediction'] as Map<String, dynamic>?;
    final timestamp = prediction?['prediction_timestamp']?.toString();
    final description = switch (_requestStatus) {
      'Running' =>
        'Sending current simulated inputs to the production model API…',
      'Succeeded' => 'Latest successful inference: $timestamp',
      'Failed' =>
        'Inference failed. No result is being shown for this request.',
      'Validation error' =>
        'Correct the highlighted inputs before requesting a prediction.',
      _ => 'Run the simulator to request a new model classification.',
    };
    final icon = switch (_requestStatus) {
      'Running' => Icons.sync,
      'Succeeded' => Icons.check_circle_outline,
      'Failed' => Icons.error_outline,
      'Validation error' => Icons.warning_amber_outlined,
      _ => Icons.hourglass_empty,
    };
    final color = switch (_requestStatus) {
      'Succeeded' => AppTheme.success,
      'Failed' || 'Validation error' => Theme.of(context).colorScheme.error,
      _ => Theme.of(context).colorScheme.secondary,
    };
    return Card(
      child: ListTile(
        leading: _loading
            ? const SizedBox.square(
                dimension: 22,
                child: CircularProgressIndicator(strokeWidth: 2),
              )
            : Icon(icon, color: color),
        title: Text('Prediction request: $_requestStatus'),
        subtitle: Text(description),
      ),
    );
  }

  Widget _askAboutPredictionAction() {
    if (_result == null || widget.onAskAboutPrediction == null) {
      return const SizedBox.shrink();
    }
    return Align(
      alignment: Alignment.centerLeft,
      child: FilledButton.icon(
        key: const ValueKey('ask-ai-about-prediction'),
        onPressed: () {
          try {
            widget.onAskAboutPrediction!(
              PredictionChatContext.fromPredictionResponse(_result!),
            );
          } on FormatException catch (error) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(content: Text(error.message)),
            );
          }
        },
        icon: const Icon(Icons.forum_outlined),
        label: const Text('Ask AI about this prediction'),
        style: FilledButton.styleFrom(
          minimumSize: const Size(300, 48),
          padding: const EdgeInsets.symmetric(horizontal: 20),
        ),
      ),
    );
  }

  Widget _resultTrainingRangeWarning() {
    final prediction = _result?['prediction'] as Map<String, dynamic>?;
    final inputs = prediction?['input_features'] as Map<String, dynamic>?;
    if (inputs == null) return const SizedBox.shrink();
    final outside = <String>[];
    for (final entry in _ranges.entries) {
      final raw = inputs[entry.key];
      if (raw is num && (raw < entry.value[0] || raw > entry.value[1])) {
        outside.add(
            '${entry.key}: $raw (training range ${entry.value[0]}–${entry.value[1]})');
      }
    }
    if (outside.isEmpty) return const SizedBox.shrink();
    return Card(
      color: AppTheme.warningSurface,
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Outside AI4I training range',
                style: TextStyle(
                    color: AppTheme.warning, fontWeight: FontWeight.w700)),
            const SizedBox(height: 4),
            const Text(
              'Inference completed, but these inputs are outside the data range used to train this model. The output may be less representative; it is not a calibrated probability or a live-machine forecast.',
              style: TextStyle(color: AppTheme.textPrimary),
            ),
            const SizedBox(height: 6),
            for (final item in outside)
              Text('• $item',
                  style: const TextStyle(color: AppTheme.textPrimary)),
          ],
        ),
      ),
    );
  }

  Widget _buildOverallAssessment() {
    final prediction = _result?['prediction'] as Map<String, dynamic>?;
    final machine = prediction?['machine_failure'] as Map<String, dynamic>?;
    final classification = machine?['classification'];
    final classified = classification == 1;
    final accent = classified ? AppTheme.danger : AppTheme.success;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Overall machine assessment',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700)),
            const SizedBox(height: 12),
            if (machine == null)
              _EmptyAssessment(
                title: _requestStatus == 'Failed'
                    ? 'Prediction failed'
                    : 'No prediction submitted',
                detail: _requestStatus == 'Failed'
                    ? 'The request did not return a model classification.'
                    : 'Run Prediction to receive a classification from the AI4I model.',
              )
            else ...[
              Row(
                children: [
                  Icon(
                    classified
                        ? Icons.warning_amber_rounded
                        : Icons.check_circle,
                    color: accent,
                    size: 30,
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      'Machine failure classification: ${classified ? 'Failure detected by classifier' : 'No failure detected by classifier'}',
                      style: const TextStyle(
                          fontSize: 17, fontWeight: FontWeight.w700),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 10),
              _rawScore(machine['score'] as num),
              const SizedBox(height: 10),
              _scoreTrack((machine['score'] as num).toDouble()),
              const SizedBox(height: 10),
              Text(
                  'Model ${prediction!['model_version']} · latest successful inference ${prediction['prediction_timestamp']}'),
              const SizedBox(height: 5),
              Text(
                'Default estimator decision: class 1 requires a score above 0.5; a tie is class 0.',
                style: Theme.of(context).textTheme.bodySmall,
              ),
              const SizedBox(height: 8),
              Text(
                'AI4I synthetic-data classification only. These simulated values are not live equipment readings and do not predict a failure date.',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildFailureModeBreakdown() {
    final prediction = _result?['prediction'] as Map<String, dynamic>?;
    final modes = prediction?['failure_modes'] as Map<String, dynamic>?;
    final unsupported =
        prediction?['unsupported_failure_modes'] as Map<String, dynamic>? ??
            const <String, dynamic>{};
    return Card(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(18, 14, 18, 14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Failure-mode breakdown',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700)),
            const SizedBox(height: 4),
            Text(
                'Classified means the loaded estimator returned class 1. The marker shows its default 0.5 class boundary; scores are uncalibrated.',
                style: Theme.of(context).textTheme.bodySmall),
            const SizedBox(height: 8),
            if (modes == null && _requestStatus == 'Failed')
              const _ModeEmptyState(
                  title: 'Prediction failed',
                  detail:
                      'No failure-mode results are available for this request.'),
            if (modes == null && _requestStatus != 'Failed')
              const _ModeEmptyState(
                  title: 'No prediction submitted',
                  detail:
                      'Run Prediction to see supported failure-mode scores.'),
            if (modes != null)
              LayoutBuilder(
                builder: (context, constraints) {
                  final columns = constraints.maxWidth >= 500 ? 2 : 1;
                  final tileWidth =
                      (constraints.maxWidth - (columns - 1) * 12) / columns;
                  return Wrap(
                    spacing: 12,
                    runSpacing: 10,
                    children: [
                      for (final entry in modes.entries)
                        SizedBox(
                          width: tileWidth,
                          child: _failureModeTile(
                              entry.key, entry.value as Map<String, dynamic>),
                        ),
                    ],
                  );
                },
              ),
            for (final entry in unsupported.entries)
              _unsupportedModeTile(entry.key, entry.value.toString()),
          ],
        ),
      ),
    );
  }

  Widget _failureModeTile(String acronym, Map<String, dynamic> result) {
    final score = (result['score'] as num).toDouble();
    final positive = result['classification'] == 1;
    final color = positive ? AppTheme.danger : AppTheme.textSecondary;
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surface,
        borderRadius: BorderRadius.circular(13),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(acronym,
                  style: const TextStyle(fontWeight: FontWeight.w700)),
              const Spacer(),
              _statusPill(positive ? 'Classified' : 'Not classified', color),
            ],
          ),
          const SizedBox(height: 2),
          Text(_failureModeNames[acronym] ?? acronym,
              style: const TextStyle(fontWeight: FontWeight.w500)),
          const SizedBox(height: 3),
          Text(
              'Raw score: ${score.toStringAsFixed(6)} · ${(score * 100).toStringAsFixed(3)}% uncalibrated',
              style: Theme.of(context).textTheme.bodySmall),
          const SizedBox(height: 6),
          _scoreTrack(score),
        ],
      ),
    );
  }

  Widget _unsupportedModeTile(String acronym, String reason) => Padding(
        padding: const EdgeInsets.only(top: 8),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 9),
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.surface,
            borderRadius: BorderRadius.circular(13),
          ),
          child: Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('$acronym — ${_failureModeNames[acronym] ?? acronym}',
                        style: const TextStyle(fontWeight: FontWeight.w600)),
                    const SizedBox(height: 3),
                    Text('No supported classifier: $reason',
                        style: Theme.of(context).textTheme.bodySmall),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              _statusPill('Unavailable', AppTheme.textSecondary),
            ],
          ),
        ),
      );

  Widget _statusPill(String label, Color color) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.14),
          borderRadius: BorderRadius.circular(999),
          border: Border.all(color: color.withValues(alpha: 0.4)),
        ),
        child: Text(label,
            style: TextStyle(
                color: color, fontSize: 12, fontWeight: FontWeight.w700)),
      );

  Widget _scoreTrack(double score) => LayoutBuilder(
        builder: (context, constraints) => SizedBox(
          height: 8,
          child: Stack(
            children: [
              Positioned.fill(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: LinearProgressIndicator(
                    value: score.clamp(0, 1),
                    backgroundColor: Theme.of(context).colorScheme.surface,
                    color: Theme.of(context).colorScheme.primary,
                    minHeight: 8,
                  ),
                ),
              ),
              Positioned(
                left: constraints.maxWidth * 0.5 - 1,
                top: 0,
                bottom: 0,
                child: Container(
                    width: 2,
                    color: AppTheme.textPrimary.withValues(alpha: .7)),
              ),
            ],
          ),
        ),
      );

  Widget _buildRagExplanation() {
    final evidence = _result!['evidence'] as List<dynamic>;
    final specific = _result!['machine_specific_evidence'] as List<dynamic>;
    final insufficient = evidence.isEmpty;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('RAG diagnostic explanation',
            style: TextStyle(fontSize: 19, fontWeight: FontWeight.bold)),
        const SizedBox(height: 10),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(insufficient ? Icons.info_outline : Icons.menu_book,
                        color: Theme.of(context).colorScheme.primary),
                    const SizedBox(width: 10),
                    Text(
                        insufficient
                            ? 'Insufficient evidence'
                            : 'Retrieved evidence',
                        style: const TextStyle(fontWeight: FontWeight.w700)),
                  ],
                ),
                const SizedBox(height: 10),
                Text(_result!['recommendation'].toString()),
                if (insufficient)
                  Padding(
                    padding: const EdgeInsets.only(top: 8),
                    child: Text(
                      'The available documents did not support a corrective action. This diagnostic evidence status is separate from the model prediction.',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ),
                if (specific.isEmpty && evidence.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.only(top: 8),
                    child: Text(
                      'No exact machine-specific record matched; the sources below are general evidence.',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ),
                if (evidence.isNotEmpty) ...[
                  const SizedBox(height: 10),
                  Text('Evidence status: ${_result!['evidence_status']}'),
                  for (var index = 0; index < evidence.length; index++)
                    ListTile(
                      dense: true,
                      contentPadding: EdgeInsets.zero,
                      leading: Text('[${index + 1}]'),
                      title: Text(
                          '${evidence[index]['source']} · ${evidence[index]['locator']}'),
                      subtitle: Text(evidence[index]['text'].toString()),
                    ),
                ],
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _rawScore(num score) => Text(
        'Raw uncalibrated score: ${score.toStringAsFixed(6)} (${(score * 100).toStringAsFixed(3)}%)',
      );
}

class _EmptyAssessment extends StatelessWidget {
  final String title;
  final String detail;

  const _EmptyAssessment({required this.title, required this.detail});

  @override
  Widget build(BuildContext context) => Container(
        width: double.infinity,
        padding: const EdgeInsets.all(18),
        decoration: BoxDecoration(
          color: Theme.of(context).colorScheme.surface,
          borderRadius: BorderRadius.circular(14),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title,
                style:
                    const TextStyle(fontSize: 17, fontWeight: FontWeight.w700)),
            const SizedBox(height: 5),
            Text(detail),
          ],
        ),
      );
}

class _ModeEmptyState extends StatelessWidget {
  final String title;
  final String detail;

  const _ModeEmptyState({required this.title, required this.detail});

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 12),
        child: Container(
          width: double.infinity,
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.surface,
            borderRadius: BorderRadius.circular(12),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(title, style: const TextStyle(fontWeight: FontWeight.w600)),
              const SizedBox(height: 4),
              Text(detail, style: Theme.of(context).textTheme.bodySmall),
            ],
          ),
        ),
      );
}
