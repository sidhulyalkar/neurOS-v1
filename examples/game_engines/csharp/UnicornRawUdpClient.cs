// neurOS Unicorn Hybrid Black raw-UDP game receiver.
//
// This file intentionally has no UnityEngine or Godot dependency. It can be
// dropped into either engine's C# project and wrapped by a MonoBehaviour/Node.
//
// Wire contract:
//   17 little-endian float32 values / 68 bytes
//   EEG1..8, ACC X/Y/Z, GYR X/Y/Z, BAT, CNT, VALID
//
// Raw UDP contains no provenance flag. Log the selected source (physical vs
// synthetic) separately in application telemetry. Do not infer it from bytes.

using System;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Threading;

namespace Neuros.Unicorn
{
    public enum UnicornStreamHealth
    {
        Stale,
        Healthy,
        Malformed,
        Invalid,
        Gap,
        Duplicate,
        OutOfOrder,
        CounterAmbiguous
    }

    public enum UnicornPacketStatus
    {
        Decodable,
        Malformed
    }

    public enum UnicornSequenceStatus
    {
        Unknown,
        First,
        Sequential,
        Gap,
        Duplicate,
        OutOfOrder,
        PrecisionAmbiguous
    }

    public sealed class UnicornRawUdpSample
    {
        public readonly float[] Values;
        public readonly int Counter;
        public readonly float BatteryLevel;
        public readonly int Validation;
        public readonly UnicornStreamHealth Health;
        public readonly int MissedPackets;
        public readonly int HealthyStreak;
        public readonly bool AuthorityAllowed;
        public readonly double ReceivedSeconds;
        public readonly string Reason;
        public readonly UnicornPacketStatus PacketStatus;
        public readonly UnicornSequenceStatus SequenceStatus;
        public readonly bool? ValidationAsserted;
        public readonly bool? CounterStepExact;

        public UnicornRawUdpSample(
            float[] values,
            int counter,
            float batteryLevel,
            int validation,
            UnicornStreamHealth health,
            int missedPackets,
            int healthyStreak,
            bool authorityAllowed,
            double receivedSeconds,
            string reason,
            UnicornPacketStatus packetStatus = UnicornPacketStatus.Decodable,
            UnicornSequenceStatus sequenceStatus = UnicornSequenceStatus.Unknown,
            bool? validationAsserted = null,
            bool? counterStepExact = null)
        {
            Values = values;
            Counter = counter;
            BatteryLevel = batteryLevel;
            Validation = validation;
            Health = health;
            MissedPackets = missedPackets;
            HealthyStreak = healthyStreak;
            AuthorityAllowed = authorityAllowed;
            ReceivedSeconds = receivedSeconds;
            Reason = reason;
            PacketStatus = packetStatus;
            SequenceStatus = sequenceStatus;
            ValidationAsserted = validationAsserted;
            CounterStepExact = counterStepExact;
        }
    }

    public sealed class UnicornRawUdpClient : IDisposable
    {
        public const int ChannelCount = 17;
        public const int PayloadBytes = 68;
        public const int BatteryIndex = 14;
        public const int CounterIndex = 15;
        public const int ValidationIndex = 16;
        public const int Float32ExactIntegerMax = 16777216; // 2^24

        private readonly object _sync = new object();
        private readonly int _port;
        private readonly double _staleAfterSeconds;
        private readonly int _recoveryPackets;
        private readonly IPAddress _bindAddress;
        private readonly Stopwatch _clock = Stopwatch.StartNew();

        private UdpClient _udp;
        private Thread _thread;
        private volatile bool _running;
        private UnicornRawUdpSample _latest;
        private int? _counterHighWater;
        private double? _lastDecodableReceiveSeconds;
        private int _healthyStreak;
        private UnicornStreamHealth _lastHealth = UnicornStreamHealth.Stale;

        public UnicornRawUdpClient(
            int port,
            double staleAfterSeconds = 0.100,
            int recoveryPackets = 3,
            string bindAddress = "127.0.0.1")
        {
            if (port < 1 || port > 65535) throw new ArgumentOutOfRangeException(nameof(port));
            if (staleAfterSeconds <= 0) throw new ArgumentOutOfRangeException(nameof(staleAfterSeconds));
            if (recoveryPackets <= 0) throw new ArgumentOutOfRangeException(nameof(recoveryPackets));
            _port = port;
            _staleAfterSeconds = staleAfterSeconds;
            _recoveryPackets = recoveryPackets;
            _bindAddress = IPAddress.Parse(bindAddress);
        }

        public void Start()
        {
            lock (_sync)
            {
                if (_running) return;
                _udp = new UdpClient(new IPEndPoint(_bindAddress, _port));
                _running = true;
                _thread = new Thread(ReceiveLoop)
                {
                    IsBackground = true,
                    Name = "neurOS-UnicornRawUdp"
                };
                _thread.Start();
            }
        }

        public void Stop()
        {
            _running = false;
            UdpClient udp;
            Thread thread;
            lock (_sync)
            {
                udp = _udp;
                thread = _thread;
                _udp = null;
                _thread = null;
            }
            try { udp?.Close(); } catch { }
            if (thread != null && thread.IsAlive && thread != Thread.CurrentThread)
            {
                thread.Join(250);
            }
        }

        public bool TryGetLatest(out UnicornRawUdpSample sample)
        {
            lock (_sync)
            {
                sample = _latest;
                return sample != null;
            }
        }

        public UnicornStreamHealth CurrentHealth
        {
            get
            {
                lock (_sync)
                {
                    if (IsStaleLocked()) return UnicornStreamHealth.Stale;
                    return _lastHealth;
                }
            }
        }

        public bool AuthorityAllowed
        {
            get
            {
                lock (_sync)
                {
                    return !IsStaleLocked()
                        && _lastHealth == UnicornStreamHealth.Healthy
                        && _healthyStreak >= _recoveryPackets;
                }
            }
        }

        private bool IsStaleLocked()
        {
            if (!_lastDecodableReceiveSeconds.HasValue) return true;
            return (_clock.Elapsed.TotalSeconds - _lastDecodableReceiveSeconds.Value) > _staleAfterSeconds;
        }

        private void ReceiveLoop()
        {
            var remote = new IPEndPoint(IPAddress.Any, 0);
            while (_running)
            {
                try
                {
                    var payload = _udp.Receive(ref remote);
                    Ingest(payload, _clock.Elapsed.TotalSeconds);
                }
                catch (ObjectDisposedException) when (!_running) { return; }
                catch (SocketException) when (!_running) { return; }
                catch
                {
                    lock (_sync)
                    {
                        _healthyStreak = 0;
                        _lastHealth = UnicornStreamHealth.Malformed;
                    }
                }
            }
        }

        public UnicornRawUdpSample Ingest(byte[] payload, double receivedSeconds)
        {
            if (payload == null) throw new ArgumentNullException(nameof(payload));
            if (double.IsNaN(receivedSeconds) || double.IsInfinity(receivedSeconds))
                throw new ArgumentOutOfRangeException(nameof(receivedSeconds));

            lock (_sync)
            {
                if (_lastDecodableReceiveSeconds.HasValue
                    && receivedSeconds - _lastDecodableReceiveSeconds.Value > _staleAfterSeconds)
                {
                    _healthyStreak = 0;
                    _lastHealth = UnicornStreamHealth.Stale;
                }

                if (payload.Length != PayloadBytes)
                {
                    return PublishMalformed(
                        receivedSeconds, $"Expected {PayloadBytes} bytes, received {payload.Length}.");
                }

                var values = new float[ChannelCount];
                for (var i = 0; i < ChannelCount; i++)
                {
                    values[i] = ReadLittleEndianSingle(payload, i * 4);
                    if (float.IsNaN(values[i]) || float.IsInfinity(values[i]))
                    {
                        return PublishMalformed(receivedSeconds, "Packet contains a non-finite value.");
                    }
                }

                var battery = values[BatteryIndex];
                var counterFloat = values[CounterIndex];
                var validationFloat = values[ValidationIndex];
                var counter = (int)Math.Round(counterFloat);
                var validation = (int)Math.Round(validationFloat);

                if (Math.Abs(counterFloat - counter) > 0.25f)
                {
                    return PublishMalformed(receivedSeconds, "Counter is not sufficiently integer-like.");
                }
                if (validation != 0 && validation != 1)
                {
                    return PublishMalformed(receivedSeconds, "Validation indicator is not binary.");
                }

                _lastDecodableReceiveSeconds = receivedSeconds;
                int missedPackets;
                bool counterStepExact;
                var sequenceStatus = ClassifySequence(counter, counterFloat, out missedPackets, out counterStepExact);
                var validationAsserted = validation == 1;
                var health = SummarizeHealth(validationAsserted, sequenceStatus);

                var sequenceOk = sequenceStatus == UnicornSequenceStatus.First
                    || sequenceStatus == UnicornSequenceStatus.Sequential;
                if (sequenceOk && validationAsserted && counterStepExact)
                {
                    _healthyStreak += 1;
                }
                else
                {
                    _healthyStreak = 0;
                }

                _lastHealth = health;
                var allowed = health == UnicornStreamHealth.Healthy
                    && _healthyStreak >= _recoveryPackets;
                var reason = BuildReason(validationAsserted, sequenceStatus, missedPackets);
                _latest = new UnicornRawUdpSample(
                    values,
                    counter,
                    battery,
                    validation,
                    health,
                    missedPackets,
                    _healthyStreak,
                    allowed,
                    receivedSeconds,
                    reason,
                    UnicornPacketStatus.Decodable,
                    sequenceStatus,
                    validationAsserted,
                    counterStepExact);
                return _latest;
            }
        }

        private UnicornSequenceStatus ClassifySequence(
            int counter,
            float counterFloat,
            out int missedPackets,
            out bool counterStepExact)
        {
            missedPackets = 0;
            counterStepExact = Math.Abs(counterFloat) <= Float32ExactIntegerMax
                && (!_counterHighWater.HasValue || Math.Abs(_counterHighWater.Value) <= Float32ExactIntegerMax);
            if (!counterStepExact)
            {
                if (!_counterHighWater.HasValue || counter > _counterHighWater.Value)
                {
                    _counterHighWater = counter;
                }
                return UnicornSequenceStatus.PrecisionAmbiguous;
            }

            if (!_counterHighWater.HasValue)
            {
                _counterHighWater = counter;
                return UnicornSequenceStatus.First;
            }

            var delta = counter - _counterHighWater.Value;
            if (delta == 0) return UnicornSequenceStatus.Duplicate;
            if (delta < 0) return UnicornSequenceStatus.OutOfOrder;

            _counterHighWater = counter;
            if (delta == 1) return UnicornSequenceStatus.Sequential;
            missedPackets = delta - 1;
            return UnicornSequenceStatus.Gap;
        }

        private static UnicornStreamHealth SummarizeHealth(
            bool validationAsserted,
            UnicornSequenceStatus sequenceStatus)
        {
            // Keep the prior compact VALID=0 behavior while preserving sequence
            // information independently on UnicornRawUdpSample.SequenceStatus.
            if (!validationAsserted) return UnicornStreamHealth.Invalid;
            switch (sequenceStatus)
            {
                case UnicornSequenceStatus.First:
                case UnicornSequenceStatus.Sequential:
                    return UnicornStreamHealth.Healthy;
                case UnicornSequenceStatus.Gap:
                    return UnicornStreamHealth.Gap;
                case UnicornSequenceStatus.Duplicate:
                    return UnicornStreamHealth.Duplicate;
                case UnicornSequenceStatus.OutOfOrder:
                    return UnicornStreamHealth.OutOfOrder;
                case UnicornSequenceStatus.PrecisionAmbiguous:
                    return UnicornStreamHealth.CounterAmbiguous;
                default:
                    return UnicornStreamHealth.Malformed;
            }
        }

        private static string BuildReason(
            bool validationAsserted,
            UnicornSequenceStatus sequenceStatus,
            int missedPackets)
        {
            var validationReason = validationAsserted ? "" : "Validation indicator is not asserted; ";
            switch (sequenceStatus)
            {
                case UnicornSequenceStatus.First:
                case UnicornSequenceStatus.Sequential:
                    return validationAsserted
                        ? "Healthy sequential validated packet."
                        : validationReason + "sequence is continuous.";
                case UnicornSequenceStatus.Gap:
                    return validationReason + $"counter gap implies {missedPackets} missing packet(s).";
                case UnicornSequenceStatus.Duplicate:
                    return validationReason + "counter repeated.";
                case UnicornSequenceStatus.OutOfOrder:
                    return validationReason + "counter arrived below the observed high-water mark.";
                case UnicornSequenceStatus.PrecisionAmbiguous:
                    return validationReason
                        + "counter exceeds float32 unit-step exactness; wrap/reset semantics are undocumented.";
                default:
                    return validationReason + "sequence status is unknown.";
            }
        }

        private UnicornRawUdpSample PublishMalformed(double receivedSeconds, string reason)
        {
            _healthyStreak = 0;
            _lastHealth = UnicornStreamHealth.Malformed;
            _latest = new UnicornRawUdpSample(
                Array.Empty<float>(),
                -1,
                float.NaN,
                -1,
                UnicornStreamHealth.Malformed,
                0,
                0,
                false,
                receivedSeconds,
                reason,
                UnicornPacketStatus.Malformed,
                UnicornSequenceStatus.Unknown,
                null,
                null);
            return _latest;
        }

        private static float ReadLittleEndianSingle(byte[] bytes, int offset)
        {
            if (BitConverter.IsLittleEndian)
            {
                return BitConverter.ToSingle(bytes, offset);
            }
            var scratch = new byte[4];
            scratch[0] = bytes[offset + 3];
            scratch[1] = bytes[offset + 2];
            scratch[2] = bytes[offset + 1];
            scratch[3] = bytes[offset];
            return BitConverter.ToSingle(scratch, 0);
        }

        public void Dispose()
        {
            Stop();
        }
    }
}
