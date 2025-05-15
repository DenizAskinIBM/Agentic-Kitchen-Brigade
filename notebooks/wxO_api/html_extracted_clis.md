## Section: ‌Extend the Default Port Address Range‌

[CLI]configure Junos OS to extend the default port address range, include thesource-portstatement at the[edit system internet-options]hierarchy level:[/CLI]

## Section: ‌Select a Fixed Source Address for Locally Generated TCP/IP Packets‌

[CLI]configure the software to select a fixed address to use as the source for locally generated IP packets, include thedefault-address-selectionstatement at the[edit system]hierarchy level:[/CLI]

## Section: ‌Rebooting and Halting a Device‌

[CLI]request system power-off[/CLI]

## Section: ‌Configure the Hostname of a Device in a Configuration Group‌

[CLI]set the hostname by using a configuration group:[/CLI]

## Section: ‌Example: Configuring the Name of the Switch, IP Address, and System ID‌

[CLI]configure the switch name, map the name to an IP address and alias, and[/CLI]
[CLI]configure a system identifier:[/CLI]

## Section: ‌Configure a DNS Name Server for Resolving Hostnames into Addresses‌

[CLI]set system management-instanceuser@host#[/CLI]
[CLI]set routing-instances mgmt_junos descriptiondescriptionuser@host#[/CLI]
[CLI]set system name-serverserver-ip-addressrouting-instance mgmt_junos[/CLI]
[CLI]set system management-instance[/CLI]
[CLI]set routing-instances mgmt_junos descriptiondescription[/CLI]
[CLI]configure the device to resolve hostnames into addresses:[/CLI]
[CLI]configure the routing instance for one of the name servers:[/CLI]
[CLI]configure the following:[/CLI]
[CLI]configure the domain name:[/CLI]

## Section: ‌Disable Reporting IP Address and Timestamps in Ping Responses‌

[CLI]configure the Routing Engine to disable the setting of therecord routeoption, include theno-ping- record-routestatement at the[edit system]hierarchy level:[/CLI]

## Section: ‌How to Rate Limit ICMPv4 and ICMPv6 Error Messages

[CLI]configure the rate limit for ICMPv4, use theicmpstatement:[/CLI]
[CLI]configure the rate limit for ICMPv6, use theicmp6statement:[/CLI]
[CLI]set the ICMP rate limit to 300 pps:[/CLI]

## Section: ‌System Alarms‌

[CLI]show system alarms4 alarms currently activeAlarm time       Class  Description2013-10-08 20:08:20 UTC Minor RE 0 /var partition usage is high 2013-10-08 20:08:20 UTC Major RE 0 /var partition is full2013-10-08 20:08:08 UTC Minor FPC 1 /var partition usage is high 2013-10-08 20:08:08 UTC Major FPC 1 /var partition is full[/CLI]
[CLI]show system alarms[/CLI]

## Section: ‌System-Wide Alarms and Alarms for Each Interface Type‌

[CLI]show system alarms[/CLI]

## Section: ‌Action

[CLI]show system uptime[/CLI]
[CLI]show system users[/CLI]
[CLI]show system storage[/CLI]
[CLI]show system processes[/CLI]
[CLI]show interfaces terse[/CLI]
[CLI]show interfaces extensive[/CLI]
[CLI]show interfacesinterface-name[/CLI]

## Section: ‌Meaning

[CLI]show system processes[/CLI]
[CLI]show system uptime[/CLI]

## Section: ‌Other Tools to Configure and Monitor Devices Running Junos OS‌

[CLI]configure and monitor devices running Junos OS:[/CLI]

## Section: ‌Configuration‌

[CLI]set interfaces et-0/0/2 passive-monitor-modeset interfaces et-0/0/2 unit 0 family inet filter input pm[/CLI]
[CLI]set interfaces et-0/0/4 passive-monitor-modeset interfaces et-0/0/4 unit 0 family inet filter input pm1set firewall family inet filter pm1 term t1 from interface et-0/0/4.0[/CLI]
[CLI]set firewall family inet filter pm1 term t1 then count c1set firewall family inet filter pm1 term t1 then routing-instance pm_inst[/CLI]
[CLI]set firewall family inet filter pm term t1 from interface et-0/0/2.0set firewall family inet filter pm term t1 then count c3set firewall family inet filter pm term t1 then routing-instance pm_inst[/CLI]
[CLI]set interfaces et-0/0/2 passive-monitor-mode[/CLI]
[CLI]set interfaces et-0/0/2 unit 0 family inet filter input pm[/CLI]
[CLI]set interfaces et-0/0/4 passive-monitor-mode[/CLI]
[CLI]set interfaces et-0/0/4 unit 0 family inet filter input pm1[/CLI]
[CLI]set firewall family inet filter pm1 term t1 from interface et-0/0/4.0[/CLI]
[CLI]set firewall family inet filter pm1 term t1 then count c1[/CLI]
[CLI]set firewall family inet filter pm1 term t1 then routing-instance pm_inst[/CLI]
[CLI]set firewall family inet filter pm term t1 from interface et-0/0/2.0[/CLI]
[CLI]set firewall family inet filter pm term t1 then count c3[/CLI]
[CLI]set firewall family inet filter pm term t1 then routing-instance pm_inst[/CLI]
[CLI]set routing-instances pm_inst instance-type virtual-router[/CLI]
[CLI]set routing-instances pm_inst interface ae0.0set routing-instances pm_inst routing-options static route 0.0.0.0/0 next-hop 198.51.100.1[/CLI]
[CLI]set interfaces xe-0/0/9:0 ether-options 802.3ad ae0set interfaces xe-0/0/9:1 ether-options 802.3ad ae0set interfaces ae0 unit 0 family inet address 198.51.100.2/24 arp 198.51.100.1 mac 00:10:94:00:00:05set routing-instances pm_inst interface ae0.0set forwarding-options enhanced-hash-key inet no-incoming-port[/CLI]
[CLI]set routing-instances pm_inst interface ae0.0[/CLI]
[CLI]set routing-instances pm_inst routing-options static route 0.0.0.0/0 next-hop 198.51.100.1[/CLI]
[CLI]set interfaces xe-0/0/9:0 ether-options 802.3ad ae0[/CLI]
[CLI]set interfaces xe-0/0/9:1 ether-options 802.3ad ae0[/CLI]
[CLI]set interfaces ae0 unit 0 family inet address 198.51.100.2/24 arp 198.51.100.1 mac 00:10:94:00:00:05[/CLI]
[CLI]set forwarding-options enhanced-hash-key inet no-incoming-port[/CLI]

## Section: Step-by-Step Procedure

[CLI]configure passive monitoring:[/CLI]
[CLI]set interfaces et-0/0/4 passive-monitor-modeset interfaces et-0/0/4 unit 0 family inet filter input pm1[/CLI]
[CLI]set interfaces et-0/0/2 passive-monitor-mode[/CLI]
[CLI]set interfaces et-0/0/2 unit 0 family inet filter input pm[/CLI]
[CLI]set interfaces et-0/0/4 passive-monitor-mode[/CLI]
[CLI]set interfaces et-0/0/4 unit 0 family inet filter input pm1[/CLI]
[CLI]set firewall family inet filter pm1 term t1 then count c1set firewall family inet filter pm1 term t1 then routing-instance pm_inst[/CLI]
[CLI]set firewall family inet filter pm term t1 from interface et-0/0/2.0set firewall family inet filter pm term t1 then count c3set firewall family inet filter pm term t1 then routing-instance pm_inst[/CLI]
[CLI]set firewall family inet filter pm1 term t1 from interface et-0/0/4.0[/CLI]
[CLI]set firewall family inet filter pm1 term t1 then count c1[/CLI]
[CLI]set firewall family inet filter pm1 term t1 then routing-instance pm_inst[/CLI]
[CLI]set firewall family inet filter pm term t1 from interface et-0/0/2.0[/CLI]
[CLI]set firewall family inet filter pm term t1 then count c3[/CLI]
[CLI]set firewall family inet filter pm term t1 then routing-instance pm_inst[/CLI]
[CLI]set routing-instances pm_inst interface ae0.0set routing-instances pm_inst routing-options static route 0.0.0.0/0 next-hop 198.51.100.1[/CLI]
[CLI]set routing-instances pm_inst instance-type virtual-router[/CLI]
[CLI]set routing-instances pm_inst interface ae0.0[/CLI]
[CLI]set routing-instances pm_inst routing-options static route 0.0.0.0/0 next-hop 198.51.100.1[/CLI]
[CLI]set interfaces xe-0/0/9:1 ether-options 802.3ad ae0set interfaces ae0 unit 0 family inet address 198.51.100.2/24 arp 198.51.100.1 mac 00:10:94:00:00:05set routing-instances pm_inst interface ae0.0[/CLI]
[CLI]set interfaces xe-0/0/9:0 ether-options 802.3ad ae0[/CLI]
[CLI]set interfaces xe-0/0/9:1 ether-options 802.3ad ae0[/CLI]
[CLI]set interfaces ae0 unit 0 family inet address 198.51.100.2/24 arp 198.51.100.1 mac 00:10:94:00:00:05[/CLI]
[CLI]set forwarding-options enhanced-hash-key inet no-incoming-port[/CLI]

## Section: Action

[CLI]show interfaces et-0/0/2[/CLI]

## Section: ‌Sample Configuration for PTX10001-36MR, PTX10004, and PTX10008 Routers‌

[CLI]set interfaces et-0/0/13 passive-monitor-mode[/CLI]
[CLI]set interfaces et-0/0/13 unit 0 family inet filter input ipv4pmFilter[/CLI]
[CLI]set interfaces et-0/0/13 unit 0 family inet6 filter input ipv6pmFilter[/CLI]
[CLI]set interfaces et-0/0/13 unit 0 family mpls filter input mplspmFilter[/CLI]
[CLI]set interfaces et-0/0/5 ether-options 802.3ad ae0[/CLI]
[CLI]set interfaces et-0/0/7 ether-options 802.3ad ae0[/CLI]
[CLI]set interfaces ae0 unit 0 family inet address 192.168.1.1/24 arp 192.168.1.10 mac 00:00:00:11:11:11[/CLI]
[CLI]set interfaces ae0 unit 0 family inet6 address 2001:db8:1::1/64 ndp 2001:db8:1::10 mac 00:00:00:11:11:11[/CLI]
[CLI]set routing-instances pm_inst routing-options rib pm_inst.inet6.0 static route 0::0/0 next-hop 2001:db8:1::10[/CLI]
[CLI]set routing-instances pm_inst routing-options static route 0.0.0.0/0 next-hop 192.168.1.10[/CLI]
[CLI]set routing-instances pm_inst instance-type virtual-router[/CLI]
[CLI]set routing-instances pm_inst interface ae0.0[/CLI]
[CLI]set firewall family inet filter ipv4pmFilter term t1 then count C1[/CLI]
[CLI]set firewall family inet filter ipv4pmFilter term t1 then routing-instance pm_inst[/CLI]
[CLI]set firewall family inet6 filter ipv6pmFilter term t2 then count C2[/CLI]
[CLI]set firewall family inet6 filter ipv6pmFilter term t2 then routing-instance pm_inst[/CLI]
[CLI]set firewall family mpls filter ipv4pmfilter term t1 then count C1[/CLI]
[CLI]set firewall family mpls filter ipv4pmfilter term t1 then routing-instance pm_inst[/CLI]
[CLI]set firewall family mpls filter ipv4pmfilter term t1 from ip-version ipv4 ip-protocol-except 255[/CLI]
[CLI]set firewall family mpls filter ipv6pmfilter term t2 then count C2[/CLI]
[CLI]set firewall family mpls filter ipv6pmfilter term t2 then routing-instance pm_inst[/CLI]
[CLI]set firewall family mpls filter ipv6pmfilter term t2 from ip-version ipv6 next-header-except 255[/CLI]

