[CLI]set system time-zone "GMT hour-offset"[/CLI]

[CLI]set system time-zone "time-zone"[/CLI]

[CLI]set system time-zone America/New_York[/CLI]

[CLI]show system time-zone[/CLI]

[CLI]set system internet-options source-port upper-limit 1024[/CLI]

[CLI]set system default-address-selection[/CLI]

[CLI]request system reboot[/CLI]

[CLI]request system reboot[/CLI]

[CLI]request system halt ?[/CLI]

[CLI]clear system reboot[/CLI]

[CLI]request system halt[/CLI]

[CLI]request system power-off[/CLI]

[CLI]clear system reboot[/CLI]

[CLI]request system halt[/CLI]

[CLI]request system power-off[/CLI]

[CLI]set apply-groups [re0 re1][/CLI]

[CLI]set apply-groups <group-name>[/CLI]

[CLI]set system host-name switch1[/CLI]

[CLI]set system static-host-mapping switch1 inet 192.168.1.77[/CLI]

[CLI]set system static-host-mapping switch1 alias sj1[/CLI]

[CLI]set system static-host-mapping switch1 sysid 1921.6800.1077[/CLI]

[CLI]show | display set[/CLI]

[CLI]set system management-instance[/CLI]

[CLI]set routing-instances mgmt_junos description description[/CLI]

[CLI]set system name-server server-ip-address routing-instance mgmt_junos[/CLI]

[CLI]set name-server 192.168.1.253[/CLI]

[CLI]set name-server 192.168.1.254[/CLI]

[CLI]set system name-server 192.168.1.253 routing-instance mgmt_junos[/CLI]

[CLI]set system domain-name company.net[/CLI]

[CLI]show system[/CLI]

[CLI]set system domain-search company.net domainone.net[/CLI]

[CLI]set apply-groups global[/CLI]

[CLI]show host host-name[/CLI]

[CLI]show host host-ip-address[/CLI]

[CLI]show host device.example.net[/CLI]

[CLI]show host 192.168.187.1[/CLI]

[CLI]set no-redirects[/CLI]

[CLI]set no-redirects-ipv6[/CLI]

[CLI]set family inet no-redirects[/CLI]

[CLI]set family inet6 no-redirects[/CLI]

[CLI]set protocols router-advertisement no-multicast-echo[/CLI]

[CLI]set no-ping-record-route[/CLI]

[CLI]set no-ping-time-stamp[/CLI]

[CLI]set ttl-expired-source-address source-address[/CLI]

[CLI]set icmp rate-limit rate-limit[/CLI]

[CLI]set icmp6 rate-limit rate-limit[/CLI]

[CLI]set chassis icmp rate-limit 300[/CLI]

[CLI]set system ddos-protection protocols exceptions mtu-exceeded bandwidth 300[/CLI]

[CLI]show system alarms[/CLI]

[CLI]show system alarms[/CLI]

[CLI]set system saved-core-files number;[/CLI]

[CLI]set system saved-core-context;[/CLI]

[CLI]show system uptime[/CLI]

[CLI]show system users[/CLI]

[CLI]show system storage[/CLI]

[CLI]show system processes[/CLI]

[CLI]show system services[/CLI]

[CLI]show system processes[/CLI]

[CLI]show system processes[/CLI]

[CLI]show system uptime[/CLI]

[CLI]show interfaces terse[/CLI]

[CLI]show interfaces extensive[/CLI]

[CLI]show interfaces interface-name[/CLI]

[CLI]show system uptime[/CLI]

[CLI]show system logs[/CLI]

[CLI]show interface diagnostics[/CLI]

[CLI]show system processes[/CLI]

[CLI]show system memory[/CLI]

[CLI]show system storage[/CLI]

[CLI]set forwarding-options enhanced-hash-key symmetric-hash[/CLI]

[CLI]set interfaces et-0/0/2 passive-monitor-mode[/CLI]

[CLI]set interfaces et-0/0/2 unit 0 family inet filter input pm[/CLI]

[CLI]set interfaces et-0/0/4 passive-monitor-mode[/CLI]

[CLI]set interfaces et-0/0/4 unit 0 family inet filter input pm1[/CLI]

[CLI]set routing-instances pm_inst instance-type virtual-router[/CLI]

[CLI]set routing-instances pm_inst interface ae0.0[/CLI]

[CLI]set routing-instances pm_inst routing-options static route 0.0.0.0/0 next-hop 198.51.100.1[/CLI]

[CLI]set routing-instances pm_inst instance-type virtual-router[/CLI]

[CLI]set routing-instances pm_inst interface ae0.0[/CLI]

[CLI]set routing-instances pm_inst routing-options static route 0.0.0.0/0 next-hop 198.51.100.1[/CLI]

[CLI]show interfaces et-0/0/2[/CLI]

[CLI]set interfaces et-0/0/13 passive-monitor-mode[/CLI]

[CLI]set interfaces et-0/0/13 passive-monitor-mode[/CLI]

[CLI]set interfaces et-0/0/13 unit 0 family inet filter input ipv4pmFilter[/CLI]

[CLI]request chassis beacon fpc slot-number on[/CLI]

[CLI]request chassis beacon fpc slot-number pic-slot slot-number port port-number[/CLI]

[CLI]request chassis beacon fpc slot-number off[/CLI]

[CLI]request chassis beacon fpc slot-number pic-slot slot-number port port-number[/CLI]

[CLI]request chassis beacon fpc slot-number on timer number-of-minutes[/CLI]

[CLI]request chassis beacon fpc slot-number pic-slot slot-number port port-number[/CLI]

[CLI]request chassis beacon fpc slot-number off[/CLI]

[CLI]request chassis beacon fpc slot-number pic-slot slot-number port port-number[/CLI]

[CLI]request chassis beacon fpc slot-number off timer number-of-minutes[/CLI]

[CLI]request chassis beacon fpc slot-number pic-slot slot-number port port-number[/CLI]