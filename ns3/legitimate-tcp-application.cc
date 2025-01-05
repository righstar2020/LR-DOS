#include <cstring>
#include "legitimate-tcp-application.h"
#include "ns3/log.h"
#include "ns3/inet-socket-address.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"
#include "ns3/socket.h"
#include "ns3/tcp-socket-factory.h"
#include "ns3/nstime.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LegitimateTPCApplication");
NS_OBJECT_ENSURE_REGISTERED (LegitimateTPCApplication);

TypeId
LegitimateTPCApplication::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::LegitimateTPCApplication")
                          .SetParent<Application> ()
                          .SetGroupName ("Applications")
                          .AddConstructor<LegitimateTPCApplication> ();
  return tid;
}

LegitimateTPCApplication::LegitimateTPCApplication ()
    : m_socket (0),
      m_peerAddress (),
      m_packetSize (0),
      m_dataRate (0),
      m_sendEvent (),
      m_running (false),
      m_totalPacketsSent (0),
      m_totalBytesReceived (0),
      m_totalDelay (0),
      m_delayedPackets (0)
{
}

LegitimateTPCApplication::~LegitimateTPCApplication ()
{
  m_socket = 0;
}

void
LegitimateTPCApplication::Setup (Address address, uint32_t packetSize, uint32_t dataRate)
{
  m_peerAddress = address;
  m_packetSize = packetSize;
  m_dataRate = dataRate;
}

void
LegitimateTPCApplication::StartApplication (void)
{
  m_running = true;
  m_startTime = Simulator::Now ();

  m_socket = Socket::CreateSocket (GetNode (), TcpSocketFactory::GetTypeId ());

  // 添加回调函数
  m_socket->SetConnectCallback (MakeCallback (&LegitimateTPCApplication::ConnectionSucceeded, this),
                                MakeCallback (&LegitimateTPCApplication::ConnectionFailed, this));
  m_socket->SetRecvCallback (MakeCallback (&LegitimateTPCApplication::HandleRead, this));

  m_socket->Connect (m_peerAddress);
  SendPacket ();
}

void
LegitimateTPCApplication::StopApplication (void)
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
LegitimateTPCApplication::SendPacket (void)
{
  if (m_running)
    {
      Ptr<Packet> packet = Create<Packet> (m_packetSize);

      // 直接使用uint64_t作为时间戳
      uint64_t timestamp = Simulator::Now ().GetNanoSeconds ();
      packet->AddAtEnd (
          Create<Packet> (reinterpret_cast<const uint8_t *> (&timestamp), sizeof (timestamp)));

      m_socket->Send (packet);
      m_totalPacketsSent++;

      // 计算下一个数据包的发送时间
      Time nextTime = Seconds (static_cast<double> (m_packetSize * 8) / m_dataRate);
      m_sendEvent = Simulator::Schedule (nextTime, &LegitimateTPCApplication::SendPacket, this);
    }
}

void
LegitimateTPCApplication::HandleRead (Ptr<Socket> socket)
{
  Ptr<Packet> packet;
  Address from;
  while ((packet = socket->RecvFrom (from)))
    {
      uint64_t sendTime;
      packet->CopyData (reinterpret_cast<uint8_t *> (&sendTime), sizeof (sendTime));

      uint64_t receiveTime = Simulator::Now ().GetNanoSeconds ();
      double delay = (receiveTime - sendTime) / 1e9; // 转换为秒

      m_totalDelay += delay;
      m_delayedPackets++;
      m_totalBytesReceived += packet->GetSize ();
    }
}

void
LegitimateTPCApplication::ConnectionSucceeded (Ptr<Socket> socket)
{
  NS_LOG_INFO ("TCP Connection succeeded");
}

void
LegitimateTPCApplication::ConnectionFailed (Ptr<Socket> socket)
{
  NS_LOG_INFO ("TCP Connection failed");
}

double
LegitimateTPCApplication::GetThroughput (void)
{
  Time now = Simulator::Now ();
  double duration = (now - m_startTime).GetSeconds ();
  if (duration > 0)
    {
      return (m_totalBytesReceived * 8.0) / duration; // 返回比特/秒
    }
  return 0;
}

double
LegitimateTPCApplication::GetAverageDelay (void)
{
  if (m_delayedPackets > 0)
    {
      return m_totalDelay / m_delayedPackets;
    }
  return 0;
}

uint64_t
LegitimateTPCApplication::GetTotalPacketsSent (void)
{
  return m_totalPacketsSent;
}

uint64_t
LegitimateTPCApplication::GetTotalBytesReceived (void)
{
  return m_totalBytesReceived;
}

} // namespace ns3