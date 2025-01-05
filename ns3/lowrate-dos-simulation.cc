#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "lowrate-dos-application.h"
#include "legitimate-tcp-application.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("LowRateDosSimulation");

Ptr<PacketSink> packetSink;

void
PrintAttackRate (Ptr<LowRateDosApplication> app)
{
  double rate = app->GetAttackRate ();
  NS_LOG_INFO ("Current attack rate: " << rate << " bytes/s");
  Simulator::Schedule (Seconds (1.0), &PrintAttackRate, app);
}

void
PrintStats ()
{
  double throughput = (packetSink->GetTotalRx () * 8.0) / Simulator::Now ().GetSeconds ();
  NS_LOG_INFO ("Legitimate user throughput: " << throughput << " bps");
  Simulator::Schedule (Seconds (1.0), &PrintStats);
}

int
main (int argc, char *argv[])
{
  Time::SetResolution (Time::NS);
  LogComponentEnable ("LowRateDosSimulation", LOG_LEVEL_INFO);

  NodeContainer nodes;
  nodes.Create (2);

  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));

  NetDeviceContainer devices;
  devices = pointToPoint.Install (nodes);

  InternetStackHelper stack;
  stack.Install (nodes);

  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0");

  Ipv4InterfaceContainer interfaces = address.Assign (devices);

  uint16_t port = 8080;

  // Set up the server application
  Address serverAddress (InetSocketAddress (Ipv4Address::GetAny (), port));
  PacketSinkHelper packetSinkHelper ("ns3::TcpSocketFactory", serverAddress);
  ApplicationContainer serverApps = packetSinkHelper.Install (nodes.Get (1));

  packetSink = DynamicCast<PacketSink> (serverApps.Get (0));

  serverApps.Start (Seconds (1.0));
  serverApps.Stop (Seconds (30.0));

  // Set up the Low-Rate DoS attack application
  Ptr<LowRateDosApplication> attackApp = CreateObject<LowRateDosApplication> ();
  attackApp->Setup (InetSocketAddress (interfaces.GetAddress (1), port), 1024, Seconds (50),
                    MilliSeconds (200), 100);
  nodes.Get (0)->AddApplication (attackApp);
  attackApp->SetStartTime (Seconds (2.0));
  attackApp->SetStopTime (Seconds (10.0));

  // Schedule periodic attack rate printing
  Simulator::Schedule (Seconds (2.0), &PrintAttackRate, attackApp);

  // 创建发送端TCP应用
  Ptr<LegitimateTPCApplication> legitApp = CreateObject<LegitimateTPCApplication> ();
  legitApp->Setup (InetSocketAddress (interfaces.GetAddress (1), port), 1024, 1000000);
  nodes.Get (0)->AddApplication (legitApp);
  legitApp->SetStartTime (Seconds (1.0));
  legitApp->SetStopTime (Seconds (30.0));

  // 定期打印统计信息
  Simulator::Schedule (Seconds (1.0), &PrintStats);

  Simulator::Run ();
  Simulator::Destroy ();
  return 0;
}