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
        OutOfOrder
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
            string reason)
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
        }
    }

    public sealed class UnicornRawUdpClient : IDisposable
    {
        public const int ChannelCount = 17;
        public const int PayloadBytes = 68;
        public const int BatteryIndex = 14;
        public const int CounterIndex = 15;
        public const int ValidationIndex = 16;

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
        private int? _lastCounter;
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
                    return PublishFault(
                        null, null, null, UnicornStreamHealth.Malformed, 0,
                        receivedSeconds, $"Expected {PayloadBytes} bytes, received {payload.Length}.");
                }

                var values = new float[ChannelCount];
                for (var i = 0; i < ChannelCount; i++)
                {
                    values[i] = ReadLittleEndianSingle(payload, i * 4);
                    if (float.IsNaN(values[i]) || float.IsInfinity(values[i]))
                    {
                        return PublishFault(
                            values, null, null, UnicornStreamHealth.Malformed, 0,
                            receivedSeconds, "Packet contains a non-finite value.");
                    }
                }

                var battery = values[BatteryIndex];
                var counterFloat = values[CounterIndex];
                var validationFloat = values[ValidationIndex];
                var counter = (int)Math.Round(counterFloat);
                var validation = (int)Math.Round(validationFloat);

                if (Math.Abs(counterFloat - counter) > 0.25f)
                {
                    return PublishFault(
                        values, null, validation, UnicornStreamHealth.Malformed, 0,
                        receivedSeconds, "Counter is not sufficiently integer-like.");
                }
                if (validation != 0 && validation != 1)
                {
                    return PublishFault(
                        values, counter, validation, UnicornStreamHealth.Malformed, 0,
                        receivedSeconds, "Validation indicator is not binary.");
                }

                _lastDecodableReceiveSeconds = receivedSeconds;
                if (validation != 1)
                {
                    _lastCounter = counter;
                    return PublishFault(
                        values, counter, validation, UnicornStreamHealth.Invalid, 0,
                        receivedSeconds, "Validation indicator is not asserted.", battery);
                }

                if (_lastCounter.HasValue)
                {
                    var delta = counter - _lastCounter.Value;
                    if (delta == 0)
                    {
                        return PublishFault(
                            values, counter, validation, UnicornStreamHealth.Duplicate, 0,
                            receivedSeconds, "Counter repeated.", battery);
                    }
                    if (delta < 0)
                    {
                        return PublishFault(
                            values, counter, validation, UnicornStreamHealth.OutOfOrder, 0,
                            receivedSeconds, "Counter moved backwards.", battery);
                    }
                    if (delta > 1)
                    {
                        _lastCounter = counter;
                        return PublishFault(
                            values, counter, validation, UnicornStreamHealth.Gap, delta - 1,
                            receivedSeconds, $"Counter advanced by {delta}.", battery);
                    }
                }

                _lastCounter = counter;
                _healthyStreak += 1;
                _lastHealth = UnicornStreamHealth.Healthy;
                var allowed = _healthyStreak >= _recoveryPackets;
                _latest = new UnicornRawUdpSample(
                    values, counter, battery, validation, UnicornStreamHealth.Healthy,
                    0, _healthyStreak, allowed, receivedSeconds,
                    "Healthy sequential validated packet.");
                return _latest;
            }
        }

        private UnicornRawUdpSample PublishFault(
            float[] values,
            int? counter,
            int? validation,
            UnicornStreamHealth health,
            int missedPackets,
            double receivedSeconds,
            string reason,
            float? battery = null)
        {
            _healthyStreak = 0;
            _lastHealth = health;
            _latest = new UnicornRawUdpSample(
                values ?? Array.Empty<float>(),
                counter ?? -1,
                battery ?? float.NaN,
                validation ?? -1,
                health,
                missedPackets,
                0,
                false,
                receivedSeconds,
                reason);
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
