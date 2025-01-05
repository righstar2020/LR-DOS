#ifndef LEGITIMATE_TCP_APPLICATION_H
#define LEGITIMATE_TCP_APPLICATION_H

#include "ns3/application.h"
#include "ns3/event-id.h"
#include "ns3/ptr.h"
#include "ns3/ipv4-address.h"
#include "ns3/traced-callback.h"
#include "ns3/socket.h"

namespace ns3 {

class LegitimateTPCApplication : public Application
{
public:
  static TypeId GetTypeId (void);
  LegitimateTPCApplication ();
  virtual ~LegitimateTPCApplication ();

  // 设置应用程序参数
  void Setup (Address address, uint32_t packetSize, uint32_t dataRate);

  // 获取统计信息
  double GetThroughput (void);
  double GetAverageDelay (void);
  uint64_t GetTotalPacketsSent (void);
  uint64_t GetTotalBytesReceived (void);

protected:
  virtual void StartApplication (void);
  virtual void StopApplication (void);

private:
  void SendPacket (void);
  void HandleRead (Ptr<Socket> socket);
  void ConnectionSucceeded (Ptr<Socket> socket);
  void ConnectionFailed (Ptr<Socket> socket);

  Ptr<Socket> m_socket; // 套接字
  Address m_peerAddress; // 目标地址
  uint32_t m_packetSize; // 数据包大小
  uint32_t m_dataRate; // 发送速率 (bps)
  EventId m_sendEvent; // 发送事件
  bool m_running; // 应用程序是否运行

  // 统计信息
  uint64_t m_totalPacketsSent; // 发送的总包数
  uint64_t m_totalBytesReceived; // 接收的总字节数
  Time m_startTime; // 开始时间
  double m_totalDelay; // 总延迟
  uint32_t m_delayedPackets; // 有延迟记录的包数
};

} // namespace ns3

#endif // LEGITIMATE_TCP_APPLICATION_H