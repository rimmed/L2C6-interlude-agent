/*
 * Copyright (c) 2013 L2jMobius
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
 * IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
package org.l2jmobius.gameserver.managers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;
import org.l2jmobius.commons.util.Rnd;
import org.l2jmobius.gameserver.config.custom.RLAgentEnvironmentConfig;
import org.l2jmobius.gameserver.data.SpawnTable;
import org.l2jmobius.gameserver.data.xml.SpawnData;
import org.l2jmobius.gameserver.geoengine.GeoEngine;
import org.l2jmobius.gameserver.model.Location;
import org.l2jmobius.gameserver.model.World;
import org.l2jmobius.gameserver.model.actor.Npc;
import org.l2jmobius.gameserver.model.actor.Player;
import org.l2jmobius.gameserver.model.actor.instance.Monster;
import org.l2jmobius.gameserver.model.spawns.Spawn;

/**
 * Farm areas of the RL agents, built from the server's own monster spawns.<br>
 * An area is a spawn territory (or, for point spawns, a grid cell) of ordinary monsters. An area suits an agent of level L when no monster is above {@code L + RLFarmLevelAbove} and most monsters are at least {@code L - RLFarmLevelBelow}.<br>
 * Every agent is assigned to one area at a time, and an area takes at most one agent per {@code RLFarmMobsPerAgent} fitting monsters, so hundreds of agents spread over the map instead of competing for the same monsters.
 */
final class RLFarmAreas
{
	private static final Logger LOGGER = Logger.getLogger(RLFarmAreas.class.getName());
	
	/** Size of the grid cell that groups point spawns (spawns without a territory). */
	private static final int CELL_SIZE = 2000;
	/** Random offset around the chosen monster, so the agent does not land inside it. */
	private static final int MONSTER_OFFSET = 100;
	/** A spot is avoided when a monster above the level window stands this close (areas of different levels can touch). */
	private static final int SAFETY_RADIUS = 1500;
	/** Live monsters of the area tried as the spot before the last one is taken anyway. */
	private static final int SPOT_ATTEMPTS = 12;
	
	/**
	 * Monsters of one area.
	 * @param key territory name or grid cell
	 * @param spawns the monster spawns of the area
	 * @param amounts monster count per level
	 * @param maxLevel highest monster level
	 */
	private record Area(String key, List<Spawn> spawns, Map<Integer, Integer> amounts, int maxLevel)
	{
		/** Monsters of the area within the level window of {@code level}, 0 when the area does not suit that level. */
		int fittingMonsters(int level)
		{
			if (maxLevel > (level + RLAgentEnvironmentConfig.RL_FARM_LEVEL_ABOVE))
			{
				return 0;
			}
			
			int total = 0;
			int fitting = 0;
			for (Map.Entry<Integer, Integer> entry : amounts.entrySet())
			{
				total += entry.getValue();
				if (entry.getKey() >= (level - RLAgentEnvironmentConfig.RL_FARM_LEVEL_BELOW))
				{
					fitting += entry.getValue();
				}
			}
			if ((fitting < RLAgentEnvironmentConfig.RL_FARM_MIN_MONSTERS) || ((fitting * 100) < (total * RLAgentEnvironmentConfig.RL_FARM_MIN_SHARE)))
			{
				return 0;
			}
			return fitting;
		}
		
		int capacity(int level)
		{
			return Math.max(1, fittingMonsters(level) / RLAgentEnvironmentConfig.RL_FARM_MOBS_PER_AGENT);
		}
	}
	
	private volatile List<Area> _areas;
	private final Map<String, Area> _byKey = new HashMap<>();
	/** Agent object id -> key of its area. */
	private final Map<Integer, String> _assignments = new ConcurrentHashMap<>();
	
	/**
	 * Picks the farm spot of an agent: its current area while that still suits its level, otherwise the least crowded suitable area.
	 * @param player the agent
	 * @return a spot next to a monster of the area, or {@code null} when no area suits the agent's level
	 */
	synchronized Location pickSpot(Player player)
	{
		final int level = player.getLevel();
		Area area = null;
		final String current = _assignments.get(player.getObjectId());
		if (current != null)
		{
			final Area assigned = _byKey.get(current);
			if ((assigned != null) && (assigned.fittingMonsters(level) > 0))
			{
				area = assigned;
			}
		}
		
		if (area == null)
		{
			area = leastCrowded(level, player.getObjectId());
			if (area == null)
			{
				_assignments.remove(player.getObjectId());
				return null;
			}
			_assignments.put(player.getObjectId(), area.key());
		}
		return spotIn(area, level);
	}
	
	/**
	 * @param player the agent
	 * @return {@code true} when the agent's area no longer suits its level (e.g. after a level-up)
	 */
	boolean outgrown(Player player)
	{
		areas();
		final String key = _assignments.get(player.getObjectId());
		final Area area = key == null ? null : _byKey.get(key);
		return (area == null) || (area.fittingMonsters(player.getLevel()) == 0);
	}
	
	void release(Player player)
	{
		_assignments.remove(player.getObjectId());
	}
	
	private Area leastCrowded(int level, int objectId)
	{
		final Map<String, Integer> crowd = new HashMap<>();
		for (Map.Entry<Integer, String> entry : _assignments.entrySet())
		{
			if (entry.getKey() != objectId)
			{
				crowd.merge(entry.getValue(), 1, Integer::sum);
			}
		}
		
		// Weighted by free places; when every suitable area is full, the least loaded one.
		final List<Area> free = new ArrayList<>();
		final List<Integer> weights = new ArrayList<>();
		Area leastLoaded = null;
		double lowestLoad = Double.MAX_VALUE;
		int totalWeight = 0;
		for (Area area : areas())
		{
			if (area.fittingMonsters(level) == 0)
			{
				continue;
			}
			
			final int capacity = area.capacity(level);
			final int agents = crowd.getOrDefault(area.key(), 0);
			if (agents < capacity)
			{
				free.add(area);
				weights.add(capacity - agents);
				totalWeight += capacity - agents;
			}
			final double load = (double) agents / capacity;
			if (load < lowestLoad)
			{
				lowestLoad = load;
				leastLoaded = area;
			}
		}
		
		if (free.isEmpty())
		{
			return leastLoaded;
		}
		
		int pick = Rnd.get(totalWeight);
		for (int i = 0; i < free.size(); i++)
		{
			pick -= weights.get(i);
			if (pick < 0)
			{
				return free.get(i);
			}
		}
		return free.get(free.size() - 1);
	}
	
	private static Location spotIn(Area area, int level)
	{
		// Next to a live monster: right height, and there is something to fight. Dead areas fall back to a spawn point.
		final List<Npc> alive = new ArrayList<>();
		for (Spawn spawn : area.spawns())
		{
			for (Npc npc : spawn.getSpawnedNpcs())
			{
				if (!npc.isDead() && npc.isSpawned())
				{
					alive.add(npc);
				}
			}
		}
		
		Location base = null;
		Collections.shuffle(alive);
		for (int i = 0; (i < alive.size()) && (i < SPOT_ATTEMPTS); i++)
		{
			final Npc npc = alive.get(i);
			base = npc.getLocation();
			if (World.getInstance().getVisibleObjectsInRange(npc, Monster.class, SAFETY_RADIUS, m -> !m.isDead() && m.getTemplate().isType("Monster") && (m.getLevel() > (level + RLAgentEnvironmentConfig.RL_FARM_LEVEL_ABOVE))).isEmpty())
			{
				break;
			}
		}
		
		if (base == null)
		{
			final Spawn spawn = area.spawns().get(Rnd.get(area.spawns().size()));
			base = spawn.isTerritoryBased() ? spawn.getSpawnTerritory().getRandomPoint() : spawn.getSpawnLocation();
		}
		
		final int x = base.getX() + Rnd.get(-MONSTER_OFFSET, MONSTER_OFFSET);
		final int y = base.getY() + Rnd.get(-MONSTER_OFFSET, MONSTER_OFFSET);
		final int z = GeoEngine.getInstance().getHeight(x, y, base.getZ());
		return Math.abs(z - base.getZ()) > 200 ? new Location(base.getX(), base.getY(), base.getZ()) : new Location(x, y, z);
	}
	
	private List<Area> areas()
	{
		List<Area> areas = _areas;
		if (areas == null)
		{
			synchronized (this)
			{
				areas = _areas;
				if (areas == null)
				{
					areas = build();
					for (Area area : areas)
					{
						_byKey.put(area.key(), area);
					}
					_areas = areas;
				}
			}
		}
		return areas;
	}
	
	private static List<Area> build()
	{
		final Map<String, List<Spawn>> grouped = new LinkedHashMap<>();
		int skipped = 0;
		for (Set<Spawn> spawns : SpawnTable.getInstance().getSpawnTable().values())
		{
			for (Spawn spawn : spawns)
			{
				if (!spawn.getTemplate().isType("Monster") || isExcluded(spawn))
				{
					skipped++;
					continue;
				}
				
				final String key = spawn.isTerritoryBased() ? "T:" + spawn.getSpawnTerritory().getName() : "C:" + Math.floorDiv(spawn.getX(), CELL_SIZE) + "," + Math.floorDiv(spawn.getY(), CELL_SIZE);
				grouped.computeIfAbsent(key, k -> new ArrayList<>()).add(spawn);
			}
		}
		
		final List<Area> areas = new ArrayList<>();
		for (Map.Entry<String, List<Spawn>> entry : grouped.entrySet())
		{
			final Map<Integer, Integer> amounts = new HashMap<>();
			int maxLevel = 0;
			for (Spawn spawn : entry.getValue())
			{
				final int level = spawn.getTemplate().getLevel();
				amounts.merge(level, Math.max(1, spawn.getAmount()), Integer::sum);
				maxLevel = Math.max(maxLevel, level);
			}
			areas.add(new Area(entry.getKey(), List.copyOf(entry.getValue()), Map.copyOf(amounts), maxLevel));
		}
		
		LOGGER.info(RLFarmAreas.class.getSimpleName() + ": " + areas.size() + " farm areas from monster spawns (" + skipped + " other spawns ignored), level window -" + RLAgentEnvironmentConfig.RL_FARM_LEVEL_BELOW + "/+" + RLAgentEnvironmentConfig.RL_FARM_LEVEL_ABOVE + ", " + RLAgentEnvironmentConfig.RL_FARM_MOBS_PER_AGENT + " monsters per agent.");
		return Collections.unmodifiableList(areas);
	}
	
	private static boolean isExcluded(Spawn spawn)
	{
		final String file = SpawnData.getInstance().getSpawnFile(spawn.getNpcSpawnTemplateId());
		if (file == null)
		{
			return false;
		}
		
		final String path = file.replace('\\', '/');
		for (String folder : RLAgentEnvironmentConfig.RL_FARM_EXCLUDED_SPAWN_FOLDERS)
		{
			if (path.contains("/" + folder + "/"))
			{
				return true;
			}
		}
		return false;
	}
}
