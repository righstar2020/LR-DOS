#ifndef LOWRATE_DOS_APPLICATION_H
#define LOWRATE_DOS_APPLICATION_H

#include "ns3/application.h"
#include "ns3/socket.h"
#include "ns3/address.h"
#include "ns3/event-id.h"
#include "ns3/nstime.h"

namespace ns3 {

class LowRateDosApplication : public Application
{
public:
  static TypeId GetTypeId (void);
  LowRateDosApplication ();
  virtual ~LowRateDosApplication ();

  void Setup (Address address, uint32_t packetSize, Time burstPeriod, Time attackPeriod,
              uint32_t burstRate);
  double GetAttackRate (void);

protected:
  virtual void StartApplication (void);
  virtual void StopApplication (void);

private:
  void SendPacket (void);
  void ControlBurst (void);
  void EndBurst (void);

  // 基础网络相关成员
  Ptr<Socket> m_socket; // 套接字
  Address m_peerAddress; // 目标地址

  // 数据包相关参数
  uint32_t m_packetSize; // 数据包大小
  uint32_t m_numPackets; // 数据包数量
  Time m_interval; // 发送间隔

  // 状态追踪
  uint32_t m_packetsSent; // 已发送的数据包数量
  EventId m_sendEvent; // 发送事件ID
  bool m_running; // 运行状态标志

  // DoS攻击相关参数
  Time m_burstPeriod; // 突发周期
  Time m_attackPeriod; // 攻击周期
  uint32_t m_burstRate; // 突发速率
  bool m_inBurst; // 突发状态标志

  // 统计信息
  uint32_t m_totalBytesSent; // 总发送字节数
  Time m_startTime; // 开始时间
};

} // namespace ns3

#endif // LOWRATE_DOS_APPLICATION_H