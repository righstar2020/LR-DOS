#include "lowrate-dos-application.h"
#include "ns3/log.h"
#include "ns3/inet-socket-address.h"
#include "ns3/simulator.h"
#include "ns3/address.h"
#include "ns3/tcp-socket-factory.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LowRateDosApplication");

NS_OBJECT_ENSURE_REGISTERED (LowRateDosApplication);

TypeId
LowRateDosApplication::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::LowRateDosApplication")
                          .SetParent<Application> ()
                          .SetGroupName ("Applications")
                          .AddConstructor<LowRateDosApplication> ();
  return tid;
}

LowRateDosApplication::LowRateDosApplication ()
    : m_socket (0),
      m_peerAddress (),
      m_packetSize (0),
      m_numPackets (0),
      m_interval (Seconds (1.0)),
      m_packetsSent (0),
      m_sendEvent (),
      m_running (false),
      m_burstPeriod (Seconds (0)),
      m_attackPeriod (Seconds (0)),
      m_burstRate (0),
      m_inBurst (false),
      m_totalBytesSent (0),
      m_startTime (Seconds (0))
{
}

LowRateDosApplication::~LowRateDosApplication ()
{
  m_socket = 0;
}
void
LowRateDosApplication::Setup (Address address, uint32_t packetSize, Time burstPeriod,
                              Time attackPeriod, uint32_t burstRate)
{
  m_peerAddress = address; // 设置目标地址
  m_packetSize = packetSize; // 设置数据包大小
  m_burstPeriod = burstPeriod; // 设置突发周期，单位为秒
  m_attackPeriod = attackPeriod; // 设置攻击周期，单位为ms
  m_burstRate = burstRate; // 设置突发速率，单位为包/秒
  m_inBurst = false; // 初始化时不在突发期
}

void
LowRateDosApplication::StartApplication (void)
{
  m_running = true;
  m_startTime = Simulator::Now ();
  m_socket = Socket::CreateSocket (GetNode (), TcpSocketFactory::GetTypeId ());
  m_socket->Connect (m_peerAddress);

  // 立即开始第一个周期
  ControlBurst ();
}

void
LowRateDosApplication::StopApplication (void)
{
  m_running = false;
  if (m_sendEvent.IsRunning ())
    {
      Simulator::Cancel (m_sendEvent);
    }
  if (m_socket)
    {
      m_socket->Close ();
    }
}

void
LowRateDosApplication::SendPacket (void)
{
  if (m_inBurst)
    {
      Ptr<Packet> packet = Create<Packet> (m_packetSize);
      m_socket->Send (packet);
      m_totalBytesSent += m_packetSize;

      Time nextSend = Seconds (1.0 / m_burstRate);
      m_sendEvent = Simulator::Schedule (nextSend, &LowRateDosApplication::SendPacket, this);
    }
}

void
LowRateDosApplication::ControlBurst (void)
{
  if (m_running)
    {
      if (!m_inBurst)
        {
          // 开始突发期
          m_inBurst = true;
          SendPacket ();

          // 计划结束突发
          Simulator::Schedule (m_burstPeriod, &LowRateDosApplication::EndBurst, this);
        }

      // 计划下一个周期
      Simulator::Schedule (m_attackPeriod, &LowRateDosApplication::ControlBurst, this);
    }
}

void
LowRateDosApplication::EndBurst (void)
{
  m_inBurst = false;
  if (m_sendEvent.IsRunning ())
    {
      Simulator::Cancel (m_sendEvent);
    }
}

double
LowRateDosApplication::GetAttackRate (void)
{
  Time now = Simulator::Now ();
  Time duration = now - m_startTime;

  if (duration.GetSeconds () == 0)
    {
      return 0.0;
    }

  double rate = static_cast<double> (m_totalBytesSent) / duration.GetSeconds ();
  return rate;
}

} // namespace ns3